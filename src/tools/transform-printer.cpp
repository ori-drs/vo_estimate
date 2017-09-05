// Tool to print rotation matrices and quats and rpys
// and to publish what bot frame computes to as to make it agree with ROS and Okvis

#include <string>
#include <iostream>
#include <fstream>

#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot_core.hpp>
#include <lcmtypes/visualization.h> // VS_OBJECT_COLLECTION_T_AXIS3D

#include <pronto_utils/pronto_vis.hpp> // visualize pt clds
#include <pronto_utils/pronto_conversions_lcm.hpp>
#include <ConciseArgs>

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

// dift filters
#include <bot_frames/bot_frames.h>
#include <bot_frames_cpp/bot_frames_cpp.hpp>


#include "pronto_lidar_filters/lidar_filters.hpp"


#include <path_util/path_util.h>

using namespace std;

struct CommandLineConfig
{
  std::string param_file;

};


class App{
  public:
    App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_);

    ~App(){}

  private:
    BotParam* botparam_;
    BotFrames* botframes_;
    bot::frames* botframes_cpp_;

    const CommandLineConfig cl_cfg_;

    void printAffine(Eigen::Affine3d q_input);


    boost::shared_ptr<lcm::LCM> lcm_recv_;
    boost::shared_ptr<lcm::LCM> lcm_pub_;

};

App::App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_) :
       lcm_recv_(lcm_recv_), lcm_pub_(lcm_pub_), cl_cfg_(cl_cfg_)
{
  botparam_ = bot_param_new_from_file(cl_cfg_.param_file.c_str());
  botframes_ = bot_frames_get_global(lcm_recv_->getUnderlyingLCM(), botparam_);
  botframes_cpp_ = new bot::frames(botframes_);



  Eigen::Affine3d imu_cam0_a;
  imu_cam0_a.matrix() << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0;
  std::cout << "imu to cam0: \n";
  printAffine(imu_cam0_a);
  std::cout << "\n";


  Eigen::Affine3d imu_cam1_a;
  imu_cam1_a.matrix() << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0;
  std::cout << "imu to cam1: \n";
  printAffine(imu_cam1_a);
  std::cout << "\n";

  Eigen::Affine3d cam0_cam1_a = imu_cam1_a.inverse() * imu_cam0_a;
  std::cout << "cam0 to cam1: \n";
  printAffine(cam0_cam1_a);
  std::cout << "\n";
  
  std::cout << cam0_cam1_a.rotation() << " rotation\n";

  Eigen::Matrix3d A = cam0_cam1_a.rotation();
  Eigen::VectorXd B(Eigen::Map<Eigen::VectorXd>(A.data(), A.cols()*A.rows()));
  std::cout << B.transpose() << " rotation as vector\n";  



  std::cout << "======================\n";
  std::cout << "======================\n";
  std::cout << "======================\n";
  Eigen::Isometry3d imu_cam0;
  botframes_cpp_->get_trans_with_utime( botframes_ ,  "imu", "CAMERA_LEFT"  , 0, imu_cam0);
  Eigen::Affine3d imu_cam0_aff = imu_cam0;
  std::cout << imu_cam0.matrix() << " imu_cam0\n\n";

  Eigen::Isometry3d imu_cam1;
  botframes_cpp_->get_trans_with_utime( botframes_ ,  "imu", "CAMERA_RIGHT"  , 0, imu_cam1);
  Eigen::Affine3d imu_cam1_aff = imu_cam1;
  std::cout << imu_cam1.matrix() << " imu_cam1\n";



  Eigen::Isometry3d cam0_cam1;
  botframes_cpp_->get_trans_with_utime( botframes_ ,  "CAMERA_LEFT", "CAMERA_RIGHT"  , 0, cam0_cam1);
  Eigen::Affine3d cam0_cam1_aff = cam0_cam1;
  std::cout << cam0_cam1.matrix() << " cam0_cam1\n";


  //Eigen::Matrix3d temp_matrix = temp_frame.matrix();
  //std::cout << temp_matrix << "\n";
  //camera_to_body.matrix();

  Eigen::Isometry3d b2s;
  botframes_cpp_->get_trans_with_utime( botframes_ ,  "ori_VELODYNE_FIXED", "body"  , 0, b2s);
  Eigen::Affine3d b2s_aff = b2s;
  std::cout << b2s.matrix() << " b2s\n";

  std::cout << print_Isometry3d(b2s) << "\n"; 
//  Eigen::Quaterniond  q = Eigen::Quaterniond(b2s.rotation()) ;
  //std::cout << q

}



void App::printAffine(Eigen::Affine3d q_input){

  std::cout << q_input.matrix() << " q_input\n";
  
  Eigen::Isometry3d q_iso;
  q_iso.setIdentity();
  q_iso.translation() = q_input.translation();
  Eigen::Quaterniond motion_R = Eigen::Quaterniond(q_input.rotation());
  q_iso.rotate( motion_R );
  
  std::cout << motion_R.w() << ", "
            << motion_R.x() << ", "
            << motion_R.y() << ", "
            << motion_R.z()
            <<  " q iso rotate quat\n";
  std::cout << q_iso.translation().transpose() <<  " q iso trans\n";

}




int main(int argc, char **argv) {
  CommandLineConfig cl_cfg;
  std::string param_file = ""; // actual file
  cl_cfg.param_file = ""; // full path to file

  ConciseArgs parser(argc, argv, "simple-fusion");
  parser.add(param_file, "c", "param_file", "Process this param file");
  parser.parse();

  cl_cfg.param_file = std::string(getConfigPath()) +'/' + std::string(param_file);
  if (param_file.empty()) { // get param from lcm
    cl_cfg.param_file = "";
  }

  std::cout << "Log filename: "              << cl_cfg.param_file         << std::endl;

  boost::shared_ptr<lcm::LCM> lcm_recv;
  boost::shared_ptr<lcm::LCM> lcm_pub;
  lcm_recv = boost::shared_ptr<lcm::LCM>(new lcm::LCM);
  lcm_pub = boost::shared_ptr<lcm::LCM>(new lcm::LCM);


  App app= App(lcm_recv, lcm_pub, cl_cfg);
  //while(0 == lcm_recv->handle());
}
