#include <string>
#include <iostream>
#include <fstream>

#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot_core.hpp>

#include <pronto_utils/pronto_vis.hpp> // visualize pt clds
#include <pronto_utils/pronto_conversions_lcm.hpp>
#include <ConciseArgs>

#include <path_util/path_util.h>

using namespace std;

struct CommandLineConfig
{
  std::string in_log_fname;
  std::string input_channel;
  int output_period;
  bool output_2d;
};

std::ofstream isam_trajectory;
std::ofstream world_trajectory;

class App{
  public:
    App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_);
    
    ~App(){
    }

  private:
    const CommandLineConfig cl_cfg_;    

    boost::shared_ptr<lcm::LCM> lcm_recv_;
    boost::shared_ptr<lcm::LCM> lcm_pub_;

    void poseHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::pose_t* msg);

    void loadPoseTrajectory(std::string database_path, std::string fname);
    void convertPoseToIsam(Isometry3dTime currentPoseT);

    pronto_vis* pc_vis_;

    Isometry3dTime previousPoseT_;

    int isam_counter_;

    std::vector<Isometry3dTime> pose_trajectory_;
};    

App::App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_) : 
       lcm_recv_(lcm_recv_), lcm_pub_(lcm_pub_), cl_cfg_(cl_cfg_), 
       previousPoseT_(0,Eigen::Isometry3d::Identity())
{

  lcm_recv_->subscribe( cl_cfg_.input_channel, &App::poseHandler,this);

  isam_trajectory.open("husky_isam_trajectory.txt");//, ios::out | ios::app | ios::binary);
  world_trajectory.open("husky_world_trajectory.txt");//, ios::out | ios::app | ios::binary);
  isam_counter_ = 0;
  cout <<"Constructed\n";


  // Mode for Filter Chain 2 mode
  /*
  pc_vis_ = new pronto_vis( lcm_pub_->getUnderlyingLCM() ); // 4 triangle, 5 triad
  pc_vis_->obj_cfg_list.push_back( obj_cfg(4004,"Pose Trajectory",5,1) );
  std::string database_path = "/home/mfallon/logs/husky/2016-07-georges-square-west/";
  std::string fname = "husky_trajectory_2_experiment.txt";
  loadPoseTrajectory(database_path, fname);
  pc_vis_->pose_collection_to_lcm_from_list(4004, pose_trajectory_);
  for (size_t i = 0; i < pose_trajectory_.size() ; i++){
     convertPoseToIsam(pose_trajectory_[i] );
  }
  cout <<"Finished converting to iSAM format\n";
  exit(-1);
  */
}



void App::loadPoseTrajectory(std::string database_path, std::string fname){

  string full_filename =  database_path + fname;
  //printf( "About to read: %s - ",full_filename.c_str());
  string line0;
  std::ifstream this_file (full_filename.c_str());
  if (this_file.is_open()){

    getline (this_file,line0);
    //cout << line0 << " is first line\n";

    while ( this_file.good() ){
      string line;
      getline (this_file,line);
      if (line.size() > 4){

        int64_t utime;

        double p[3], q[4], r[3];
        sscanf(line.c_str(), "%ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&utime,
            &(p[0]), &(p[1]), &(p[2]), // position
            &(q[0]), &(q[1]), &(q[2]),  &(q[3]), // quat wxyz
            &(r[0]), &(r[1]), &(r[2])); // rpy (unused)

        Eigen::Isometry3d pose;
        pose.setIdentity();
        pose.translation() << p[0],p[1],p[2];
        Eigen::Quaterniond quat(q[0],q[1],q[2],q[3]);
        pose.rotate(quat);
        Isometry3dTime poseT = Isometry3dTime(utime, pose);

        //cout << utime << " | ";
        //cout << "p: " << p[0] << " " << p[1] << " " << p[2] << " | ";
        //cout << "q: " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " | ";
        //cout << "r: " << r[0] << " " << r[1] << " " << r[2] << "\n";

        pose_trajectory_.push_back(poseT);
      }
    }
    this_file.close();
  } else{
    printf( "Unable to open trajectory file\n%s",fname.c_str());
    exit(-1);
    return;
  }
  cout << "Loaded pose trajectory of length " << pose_trajectory_.size() << "\n";
}



void App::poseHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::pose_t* msg){
  Isometry3dTime currentPoseT = pronto::getPoseAsIsometry3dTime(msg);
  convertPoseToIsam(currentPoseT);
}

// 0 will skip first 10 frames and print the 10th
// 1 will print the first frame
int counter =0; 
void App::convertPoseToIsam(Isometry3dTime currentPoseT){
  counter++;

  if (previousPoseT_.utime == 0){
    std::cout << "initializing pose\n";
    previousPoseT_ = currentPoseT;
    return;
  }

  //std::cout << msg->utime << "\n";
  if (counter% cl_cfg_.output_period > 0){
    return;
  }

  // 1. Calculate the relative distance travelled since the last output:
  Eigen::Isometry3d deltaPose = previousPoseT_.pose.inverse() * currentPoseT.pose ;  
  Isometry3dTime deltaPoseT = Isometry3dTime(currentPoseT.utime, deltaPose);
  //bot_core::pose_t msg_out = pronto::getIsometry3dAsBotPose(deltaPoseT.pose, deltaPoseT.utime);
  //lcm_->publish(std::string("POSE_BODY_DELTA"),&msg_out);
  std::cout << currentPoseT.utime << " " << deltaPoseT.pose.translation().transpose() << "\n";


  // 2. Output the iSAM constraints (in either 2D or 3D) and a trajectory 
  isam_counter_++;
  if (cl_cfg_.output_2d){
  double delta_rpy[3];
  quat_to_euler( Eigen::Quaterniond( deltaPoseT.pose.rotation() ) , delta_rpy[0], delta_rpy[1], delta_rpy[2]);
  isam_trajectory << "ODOMETRY " << (isam_counter_-1) << " " << isam_counter_ << " " 
                  << deltaPoseT.pose.translation().x() << " "<< deltaPoseT.pose.translation().y() << " "
                  << delta_rpy[2] << " "
                  << " 50 0 0 50 0 100\n";
  isam_trajectory.flush();


  double current_rpy[3];
  quat_to_euler( Eigen::Quaterniond( currentPoseT.pose.rotation() ) , current_rpy[0], current_rpy[1], current_rpy[2]);
  world_trajectory << currentPoseT.utime  << " " << (isam_counter_) << " "
                  << currentPoseT.pose.translation().x() << " " << currentPoseT.pose.translation().y() << " "
                  << current_rpy[2] << "\n";
  world_trajectory.flush();
}else{
  double delta_rpy[3];
  quat_to_euler( Eigen::Quaterniond( deltaPoseT.pose.rotation() ) , delta_rpy[0], delta_rpy[1], delta_rpy[2]);
  isam_trajectory << "EDGE3 " << (isam_counter_-1) << " " << isam_counter_ << " " 
                  << deltaPoseT.pose.translation().x() << " " << deltaPoseT.pose.translation().y() << " " << deltaPoseT.pose.translation().z() << " "
                  << delta_rpy[0] << " " << delta_rpy[1] << " " << delta_rpy[2] << " "
                  << " 10 0 0 0 0 0 10 0 0 0 0 10 0 0 0 100 0 0 100 0 25\n";
  //                << " xx xyxzr p y yy z r p y zz r p y rr  p y pp  y y"
  isam_trajectory.flush();



  double current_rpy[3];
  Eigen::Quaterniond current_quat = Eigen::Quaterniond( currentPoseT.pose.rotation() );
  quat_to_euler( current_quat  , current_rpy[0], current_rpy[1], current_rpy[2]);
  world_trajectory << currentPoseT.utime  << " " << (isam_counter_) << " "
                  << currentPoseT.pose.translation().x() << " " << currentPoseT.pose.translation().y() << " " << currentPoseT.pose.translation().z() << " "
                  << current_quat.w() << " " << current_quat.x() << " " << current_quat.y() << " " << current_quat.z() << " "
                  << current_rpy[0] << " " << current_rpy[1] << " " << current_rpy[2] << "\n";
  world_trajectory.flush();
}


  previousPoseT_ = currentPoseT;
}



int main(int argc, char **argv){
  CommandLineConfig cl_cfg;
  cl_cfg.input_channel = "POSE_BODY";  
  cl_cfg.in_log_fname = "";
  cl_cfg.output_period = 1; // don't skip any. another option would be 10 - skip every 10
  cl_cfg.output_2d = false;
  double processing_rate = 1; // real time

  ConciseArgs parser(argc, argv, "simple-fusion");
  parser.add(cl_cfg.input_channel, "i", "input_channel", "input_channel - POSE_BODY typically");
  parser.add(cl_cfg.in_log_fname, "L", "in_log_fname", "Process this log file");
  parser.add(processing_rate, "pr", "processing_rate", "Processing Rate from a log [0=ASAP, 1=realtime]");  
  parser.add(cl_cfg.output_period, "o", "output_period", "Period between constraints output");  
  parser.add(cl_cfg.output_2d, "d", "output_2d", "output 2d isam constraints");  
  parser.parse();
  
  //
  bool running_from_log = !cl_cfg.in_log_fname.empty();
  boost::shared_ptr<lcm::LCM> lcm_recv;
  boost::shared_ptr<lcm::LCM> lcm_pub;
  if (running_from_log) {
    printf("running from log file: %s\n", cl_cfg.in_log_fname.c_str());
    std::stringstream lcmurl;
    lcmurl << "file://" << cl_cfg.in_log_fname << "?speed=" << processing_rate;
    lcm_recv = boost::shared_ptr<lcm::LCM>(new lcm::LCM(lcmurl.str()));
    if (!lcm_recv->good()) {
      fprintf(stderr, "Error couldn't load log file %s\n", lcmurl.str().c_str());
      exit(1);
    }
  }
  else {
    lcm_recv = boost::shared_ptr<lcm::LCM>(new lcm::LCM);
  }
  lcm_pub = boost::shared_ptr<lcm::LCM>(new lcm::LCM);

  App app= App(lcm_recv, lcm_pub, cl_cfg);
  while(0 == lcm_recv->handle());
}
