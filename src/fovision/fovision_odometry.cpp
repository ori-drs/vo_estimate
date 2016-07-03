// A VO-based non-probablistic state estimator for the multisense
// - occasionally uses IMU to avoid orientation drift
// - when VO fails extrapolate using previous vision lin rate and imu rot rates

// For IMU orientation integration:
// Estimate is maintained in the body frame which is assumed to be
// Forward-Left-Up such at roll and pitch can be isolated from yaw.


#include <zlib.h>
#include <lcm/lcm-cpp.hpp>
#include <bot_param/param_client.h>
#include <bot_frames/bot_frames.h>
#include <bot_frames_cpp/bot_frames_cpp.hpp>

#include <lcmtypes/bot_core.hpp>
#include <lcmtypes/multisense.hpp>

#include "drcvision/voconfig.hpp"
#include "drcvision/vofeatures.hpp"
#include "drcvision/voestimator.hpp"
#include "fovision.hpp"

#include <pronto_utils/pronto_vis.hpp> // visualize pt clds
#include <image_io_utils/image_io_utils.hpp> // to simplify jpeg/zlib compression and decompression
#include <ConciseArgs>

/// For Forward Kinematics from body to head:
#include "urdf/model.h"
#include "kdl/tree.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "forward_kinematics/treefksolverposfull_recursive.hpp"
#include <model-client/model-client.hpp>
#include <path_util/path_util.h>

#include <opencv/cv.h> // for disparity 

using namespace std;
using namespace cv; // for disparity ops

struct CommandLineConfig
{
  std::string camera_config; // which block from the cfg to read
  // 0 none, 1 at init, 2 rpy, 2 rp only
  int fusion_mode;
  bool feature_analysis;
  int feature_analysis_publish_period; // number of frames between publishing the point features 
  std::string output_extension;
  std::string output_signal;
  std::string input_channel;
  bool verbose;
  int correction_frequency;
  int atlas_version;
  std::string imu_channel;
  std::string in_log_fname;
  std::string param_file;
};

class StereoOdom{
  public:
    StereoOdom(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_);
    
    ~StereoOdom(){
      free (left_buf_);
      free(right_buf_);
    }

  private:
    const CommandLineConfig cl_cfg_;    
    
    int image_size_; // just the resolution of the image
    uint8_t* left_buf_;
    uint8_t* right_buf_;
    mutable std::vector<float> disparity_buf_; // Is mutable necessary?
    uint8_t* rgb_buf_ ;
    uint8_t* decompress_disparity_buf_;    
    image_io_utils*  imgutils_;    
    
    int64_t utime_cur_, utime_prev_;

    boost::shared_ptr<lcm::LCM> lcm_recv_;
    boost::shared_ptr<lcm::LCM> lcm_pub_;
    BotParam* botparam_;
    BotFrames* botframes_;
    bot::frames* botframes_cpp_;
    voconfig::KmclConfiguration* config_;

    // Vision and Estimation
    FoVision* vo_;
    VoFeatures* features_;
    uint8_t* left_buf_ref_; // copies of the reference images - probably can be extracted from fovis directly
    int64_t ref_utime_;
    Eigen::Isometry3d ref_camera_pose_; // [pose of the camera when the reference frames changed
    bool changed_ref_frames_;
    VoEstimator* estimator_;
    void featureAnalysis();
    void updateMotion();

    void multisenseHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::images_t* msg);
    void multisenseLDHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::images_t* msg);
    void multisenseLRHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::images_t* msg);

    // Kinematics
    boost::shared_ptr<ModelClient> model_;
    boost::shared_ptr<KDL::TreeFkSolverPosFull_recursive> fksolver_;

    // IMU
    bool pose_initialized_; // initalized from VO
    int imu_counter_;
    void microstrainHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::ins_t* msg);
    Eigen::Quaterniond imuOrientationToRobotOrientation(const bot_core::ins_t *msg);
    void fuseInterial(Eigen::Quaterniond local_to_body_orientation_from_imu, int64_t utime);
    Eigen::Isometry3d body_to_imu_, imu_to_camera_;
    
    // previous successful vo estimates as rates:
    Eigen::Vector3d vo_velocity_linear_;
    Eigen::Vector3d vo_velocity_angular_;
    Eigen::Vector3d camera_linear_velocity_from_imu_; // in camera frame
    Eigen::Vector3d camera_angular_velocity_from_imu_; // in camera frame
    Eigen::Vector3d camera_angular_velocity_from_imu_alpha_; // in camera frame
};    

StereoOdom::StereoOdom(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_) : 
       lcm_recv_(lcm_recv_), lcm_pub_(lcm_pub_), cl_cfg_(cl_cfg_), utime_cur_(0), utime_prev_(0), 
       ref_utime_(0), changed_ref_frames_(false)
{
  // Set up frames and config:
  if (cl_cfg_.param_file.empty()) {
    std::cout << "Get param from LCM\n";
    botparam_ = bot_param_get_global(lcm_recv_->getUnderlyingLCM(), 0);
  } else {
    std::cout << "Get param from file\n";
    botparam_ = bot_param_new_from_file(cl_cfg_.param_file.c_str());
  }
  if (botparam_ == NULL) {
    exit(1);
  }
  botframes_= bot_frames_get_global(lcm_recv_->getUnderlyingLCM(), botparam_);
  botframes_cpp_ = new bot::frames(botframes_);

  /*
  model_ = boost::shared_ptr<ModelClient>(new ModelClient(lcm_recv_->getUnderlyingLCM(), 0));
  KDL::Tree tree;
  if (!kdl_parser::treeFromString( model_->getURDFString() ,tree)){
    cerr << "ERROR: Failed to extract kdl tree from xml robot description" << endl;
    exit(-1);
  }
  fksolver_ = boost::shared_ptr<KDL::TreeFkSolverPosFull_recursive>(new KDL::TreeFkSolverPosFull_recursive(tree));
  //last_atlas_state_msg_.utime = 0;
  */

  config_ = new voconfig::KmclConfiguration(botparam_, cl_cfg_.camera_config);
  boost::shared_ptr<fovis::StereoCalibration> stereo_calibration_;
  stereo_calibration_ = boost::shared_ptr<fovis::StereoCalibration>(config_->load_stereo_calibration());

  // Allocate various buffers:
  image_size_ = stereo_calibration_->getWidth() * stereo_calibration_->getHeight();
  left_buf_ = (uint8_t*) malloc(3*image_size_);
  right_buf_ = (uint8_t*) malloc(3*image_size_);
  left_buf_ref_ = (uint8_t*) malloc(3*image_size_); // used of feature output 
  rgb_buf_ = (uint8_t*) malloc(10*image_size_ * sizeof(uint8_t)); 
  decompress_disparity_buf_ = (uint8_t*) malloc( 4*image_size_*sizeof(uint8_t));  // arbitary size chosen..
  imgutils_ = new image_io_utils( lcm_pub_, stereo_calibration_->getWidth(), 2*stereo_calibration_->getHeight()); // extra space for stereo tasks

  vo_ = new FoVision(lcm_pub_ , stereo_calibration_);
  features_ = new VoFeatures(lcm_pub_, stereo_calibration_->getWidth(), stereo_calibration_->getHeight() );
  estimator_ = new VoEstimator(lcm_pub_ , botframes_, cl_cfg_.output_extension );

  Eigen::Isometry3d init_pose;
  init_pose = Eigen::Isometry3d::Identity();
  init_pose.translation() = Eigen::Vector3d(0,0,1.65); // nominal head height
  estimator_->setBodyPose(init_pose);

  // IMU:
  pose_initialized_=false;
  imu_counter_=0;
  lcm_recv_->subscribe( cl_cfg_.input_channel, &StereoOdom::multisenseHandler,this);
  lcm_recv_->subscribe( cl_cfg_.imu_channel, &StereoOdom::microstrainHandler,this);
  // This assumes the imu to body frame is fixed, need to update if the neck is actuated
  botframes_cpp_->get_trans_with_utime( botframes_ ,  "body", "imu", 0, body_to_imu_);
  botframes_cpp_->get_trans_with_utime( botframes_ ,  "imu", "CAMERA_LEFT", 0, imu_to_camera_);


  cout <<"StereoOdom Constructed\n";
}


int counter =0;
void StereoOdom::featureAnalysis(){

  /// Incremental Feature Output:
  if (counter% cl_cfg_.feature_analysis_publish_period == 0 ){
    features_->setFeatures(vo_->getMatches(), vo_->getNumMatches() , utime_cur_);
    features_->setCurrentImage(left_buf_);
    //features_->setCurrentImages(left_buf_, right_buf_);
    features_->setCurrentCameraPose( estimator_->getCameraPose() );
    features_->doFeatureProcessing(1); // 1 = send the FEATURES_CUR
  }
  
  /// Reference Feature Output: ///////////////////////////////////////////////
  // Check we changed reference frame last iteration, if so output the set of matching inliers:
  if (changed_ref_frames_) {
    if (ref_utime_ > 0){ // skip the first null image
      if(vo_->getNumMatches() > 200){ // if less than 50 features - dont bother writing
      // was:      if(featuresA.size() > 50){ // if less than 50 features - dont bother writing
        cout << "ref frame from " << utime_prev_ << " at " << utime_cur_ <<  " with " <<vo_->getNumMatches()<<" matches\n";
        features_->setFeatures(vo_->getMatches(), vo_->getNumMatches() , ref_utime_);
        features_->setReferenceImage(left_buf_ref_);
        features_->setReferenceCameraPose( ref_camera_pose_ );
        features_->doFeatureProcessing(0); // 0 = send the FEATURES_REF
      }
    }
    changed_ref_frames_=false;
  }

  if (vo_->getChangeReferenceFrames()){ // If we change reference frame, note the change for the next iteration.
    ref_utime_ = utime_cur_;
    ref_camera_pose_ = estimator_->getCameraPose(); // publish this pose when the 
    // TODO: only copy gray data if its grey
    std::copy( left_buf_ , left_buf_ + 3*image_size_  , left_buf_ref_); // Keep the image buffer to write with the features:
    changed_ref_frames_=true;
  }
  counter++;
}

void StereoOdom::updateMotion(){
  // 1. Estimate the motion using VO
  Eigen::Isometry3d delta_camera;
  Eigen::MatrixXd delta_cov;
  fovis::MotionEstimateStatusCode delta_status;
  vo_->getMotion(delta_camera, delta_cov, delta_status );
  vo_->fovis_stats();

  // 2. If successful cache the rates
  //    otherwise extrapolate the previous rate
  if (delta_status == fovis::SUCCESS){
    // Get the 1st order rates:
    double dt = (double) ((utime_cur_ - utime_prev_)*1E-6);
    vo_velocity_linear_ = Eigen::Vector3d( delta_camera.translation().x() / dt ,
                                           delta_camera.translation().y() / dt ,
                                           delta_camera.translation().z() / dt);
    double rpy[3];
    quat_to_euler(  Eigen::Quaterniond(delta_camera.rotation()) , rpy[0], rpy[1], rpy[2]);
    vo_velocity_angular_ = Eigen::Vector3d( rpy[0]/dt , rpy[1]/dt , rpy[2]/dt);

    ////std::cout << vo_velocity_linear_.transpose() << " vo linear\n";
    //Eigen::Vector3d head_velocity_linear = estimator_->getBodyLinearRate();
    //std::cout << head_velocity_linear.transpose() << " vo linear head\n";

    //std::cout << vo_velocity_angular_.transpose() << " vo angular\n";
    //Eigen::Vector3d head_velocity_angular = estimator_->getBodyRotationRate();
    //std::cout << head_velocity_angular.transpose() << " vo angular head\n";

  }else{
    double dt = (double) ((utime_cur_ - utime_prev_)*1E-6);
    std::cout << "failed VO\n";
    if (fabs(dt) > 0.2){
      delta_camera.setIdentity();
      std::cout << "================ Unexpected jump: " << dt << " sec. Not extrapolating ==========\n";
    }else{

      // This orientation is not mathematically correct:
      std::cout << dt << " sec | "
                << vo_velocity_linear_.transpose()   << " m/s | "
                << camera_angular_velocity_from_imu_.transpose() << " r/s to be extrapolated\n";

      // Use IMU rot_rates to extrapolate rotation: This is mathematically incorrect:
      Eigen::Quaterniond extrapolated_quat = euler_to_quat( camera_angular_velocity_from_imu_[0]*dt, camera_angular_velocity_from_imu_[1]*dt, camera_angular_velocity_from_imu_[2]*dt);

      delta_camera.setIdentity();
      delta_camera.translation().x() = vo_velocity_linear_[0] * dt;
      delta_camera.translation().y() = vo_velocity_linear_[1] * dt;
      delta_camera.translation().z() = vo_velocity_linear_[2] * dt;
      delta_camera.rotate(extrapolated_quat);
    }

  }
  if (cl_cfg_.verbose){
    std::stringstream ss2;
    print_Isometry3d(delta_camera, ss2);
    std::cout << "camera: " << ss2.str() << " code" << (int) delta_status << "\n";
  }

  // 3. Update the motion estimation:
  estimator_->updatePosition(utime_cur_, utime_prev_, delta_camera);
}

int pixel_convert_8u_rgb_to_8u_gray (uint8_t *dest, int dstride, int width,
        int height, const uint8_t *src, int sstride)
{
  int i, j;
  for (i=0; i<height; i++) {
    uint8_t *drow = dest + i * dstride;
    const uint8_t *srow = src + i * sstride;
    for (j=0; j<width; j++) {
      drow[j] = 0.2125 * srow[j*3+0] +
        0.7154 * srow[j*3+1] +
        0.0721 * srow[j*3+2];
    }
  }
  return 0;
}





void StereoOdom::multisenseHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::images_t* msg){

  if (!pose_initialized_){
    return;
  }

  utime_prev_ = utime_cur_;
  utime_cur_ = msg->utime;

  // Detect the image stream and process accordingly
  if ( (msg->image_types[0] ==  bot_core::images_t::LEFT) &&
       (msg->image_types[1] ==  bot_core::images_t::RIGHT) ) {

    multisenseLRHandler(rbuf, channel, msg);
    vo_->doOdometry(left_buf_,right_buf_, msg->utime);
  }else if( (msg->image_types[0] ==  bot_core::images_t::LEFT) &&
       (msg->image_types[1] ==  bot_core::images_t::DISPARITY_ZIPPED) ) {

    multisenseLDHandler(rbuf, channel, msg);
    vo_->doOdometry(left_buf_,disparity_buf_.data(), msg->utime );
  }else{
    std::cout << "StereoOdom::multisenseHandler | image pairings not understood\n";
    return;
  }
  updateMotion();

  if(cl_cfg_.feature_analysis)
    featureAnalysis();

}


void StereoOdom::multisenseLDHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::images_t* msg){

  int w = msg->images[0].width;
  int h = msg->images[0].height;

  if (msg->images[0].pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_RGB ){
    rgb_buf_ = (uint8_t*) msg->images[0].data.data();
  }else if (msg->images[0].pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_GRAY ){
    rgb_buf_ = (uint8_t*) msg->images[0].data.data();
  }else if (msg->images[0].pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG ){
    jpeg_decompress_8u_rgb ( msg->images[0].data.data(), msg->images[0].size, rgb_buf_, w, h, w* 3);
    pixel_convert_8u_rgb_to_8u_gray(  left_buf_, w, w, h, rgb_buf_,  w*3);
  }else{
    std::cout << "StereoOdom image type not understood\n";
    exit(-1);
  }

  // TODO: support other modes (as in the renderer)
  if (msg->image_types[1] == bot_core::images_t::DISPARITY_ZIPPED) {
    unsigned long dlen = w*h*2;
    uncompress(decompress_disparity_buf_ , &dlen, msg->images[1].data.data(), msg->images[1].size);
  } else{
    std::cout << "StereoOdom depth type not understood\n";
    exit(-1);
  }

  // Convert Carnegie disparity format into floating point disparity. Store in local buffer
  Mat disparity_orig_temp = Mat::zeros(h,w,CV_16UC1); // h,w
  disparity_orig_temp.data = (uchar*) decompress_disparity_buf_;   // ... is a simple assignment possible?
  cv::Mat_<float> disparity_orig(h, w);
  disparity_orig = disparity_orig_temp;
  disparity_buf_.resize(h * w);
  cv::Mat_<float> disparity(h, w, &(disparity_buf_[0]));
  disparity = disparity_orig / 16.0;

  return;
}


void StereoOdom::multisenseLRHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::images_t* msg){

  int w = msg->images[0].width;
  int h = msg->images[0].height;

  if (msg->images[0].pixelformat != msg->images[1].pixelformat){
    std::cout << "Pixel formats not identical, not supported\n";
    exit(-1);
  }

  switch (msg->images[0].pixelformat) {
    case BOT_CORE_IMAGE_T_PIXEL_FORMAT_GRAY:
      memcpy(left_buf_,  msg->images[0].data.data() , msg->images[0].size);
      memcpy(right_buf_,  msg->images[1].data.data() , msg->images[1].size);
      break;
    case BOT_CORE_IMAGE_T_PIXEL_FORMAT_RGB:
      // image came in as raw RGB buffer.  convert to grayscale:
      pixel_convert_8u_rgb_to_8u_gray(  left_buf_ , w, w, h, msg->images[0].data.data(),  w*3);
      pixel_convert_8u_rgb_to_8u_gray(  right_buf_, w, w, h, msg->images[1].data.data(),  w*3);
      break;
    case BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG:
      // not sure why but the same setting seem to work for both jpeg compressed color (left) and grey (right):
      jpeg_decompress_8u_gray(msg->images[0].data.data(), msg->images[0].size,
                              left_buf_, w, h, w);
      jpeg_decompress_8u_gray(msg->images[1].data.data(), msg->images[1].size,
                              right_buf_, w, h, w);
      break;
    default:
      std::cout << "Unrecognized image format\n";
      exit(-1);
      break;
  }

  return;
}


// Transform the Microstrain IMU orientation into the body frame:
Eigen::Quaterniond StereoOdom::imuOrientationToRobotOrientation(const bot_core::ins_t *msg){
  Eigen::Quaterniond m(msg->quat[0],msg->quat[1],msg->quat[2],msg->quat[3]);
  Eigen::Isometry3d motion_estimate;
  motion_estimate.setIdentity();
  motion_estimate.translation() << 0,0,0;
  motion_estimate.rotate(m);

  // rotate IMU orientation so that look vector is +X, and up is +Z
  // TODO: do this with rotation matrices for efficiency:
  Eigen::Isometry3d body_pose_from_imu = motion_estimate * body_to_imu_;
  Eigen::Quaterniond body_orientation_from_imu(body_pose_from_imu.rotation());
  // For debug:
  //estimator_->publishPose(msg->utime, "POSE_BODY" , motion_estimate_out, Eigen::Vector3d::Identity() , Eigen::Vector3d::Identity());
  //estimator_->publishPose(msg->utime, "POSE_BODY_ALT" , motion_estimate, Eigen::Vector3d::Identity() , Eigen::Vector3d::Identity());

  return body_orientation_from_imu;
}

void StereoOdom::fuseInterial(Eigen::Quaterniond local_to_body_orientation_from_imu, int64_t utime){

  if (cl_cfg_.fusion_mode==0) // Got IMU measurement - not incorporating them.
    return;
  
  if (!pose_initialized_){
    std::vector<int> init_modes = {1,2,3};
    if(std::find(init_modes.begin(), init_modes.end(), cl_cfg_.fusion_mode) != init_modes.end()) {
    //if((cl_cfg_.fusion_mode ==1) ||(cl_cfg_.fusion_mode ==2)  ){
      Eigen::Isometry3d init_pose;
      init_pose.setIdentity();
      init_pose.translation() << 0,0,0;
      init_pose.rotate( local_to_body_orientation_from_imu );
      estimator_->setBodyPose(init_pose);
      pose_initialized_ = true;
      cout << "got first IMU measurement\n";
      return;
    }
  }
  
  if((cl_cfg_.fusion_mode ==2) ||(cl_cfg_.fusion_mode ==3) ){
    if (imu_counter_== cl_cfg_.correction_frequency){
      // Every X frames: replace the pitch and roll with that from the IMU
      // convert the camera pose to head frame
      // extract xyz and yaw from head frame
      // extract pitch and roll from imu (in head frame)
      // combine, convert to camera frame... set as pose

      // 1. Get the currently estimated head pose and its rpy
      Eigen::Isometry3d local_to_head = estimator_->getBodyPose();// _local_to_camera *cam2head;
      std::stringstream ss2;
      print_Isometry3d(local_to_head, ss2);
      double rpy[3];
      quat_to_euler(  Eigen::Quaterniond(local_to_head.rotation()) , rpy[0], rpy[1], rpy[2]);
      if (cl_cfg_.verbose){
        std::cout << "local_to_head: " << ss2.str() << " | "<< 
          rpy[0]*180/M_PI << " " << rpy[1]*180/M_PI << " " << rpy[2]*180/M_PI << "\n";        
      }
        
      // 2. Get the IMU orientated RPY:
      double rpy_imu[3];
      quat_to_euler( local_to_body_orientation_from_imu , 
                      rpy_imu[0], rpy_imu[1], rpy_imu[2]);
      if (cl_cfg_.verbose){
        std::cout <<  rpy_imu[0]*180/M_PI << " " << rpy_imu[1]*180/M_PI << " " << rpy_imu[2]*180/M_PI << " rpy_imu\n";        
        cout << "IMU correction | roll pitch | was: "
            << rpy[0]*180/M_PI << " " << rpy[1]*180/M_PI << " | now: "
            << rpy_imu[0]*180/M_PI << " " << rpy_imu[1]*180/M_PI << "\n";
      }
      
      // 3. Merge the two orientation estimates:
      Eigen::Quaterniond revised_local_to_body_quat;
      if (cl_cfg_.fusion_mode==2){ // rpy:
        revised_local_to_body_quat = local_to_body_orientation_from_imu;
      }else{  // pitch and roll from IMU, yaw from VO:
        revised_local_to_body_quat = euler_to_quat( rpy_imu[0], rpy_imu[1], rpy[2]);
      }
      ///////////////////////////////////////
      Eigen::Isometry3d revised_local_to_head;
      revised_local_to_head.setIdentity();
      revised_local_to_head.translation() = local_to_head.translation();
      revised_local_to_head.rotate(revised_local_to_body_quat);
      
      // 4. Set the Head pose using the merged orientation:
      if (cl_cfg_.verbose){
        std::stringstream ss4;
        print_Isometry3d(revised_local_to_head, ss4);
        quat_to_euler(  Eigen::Quaterniond(revised_local_to_head.rotation()) , rpy[0], rpy[1], rpy[2]);
        std::cout << "local_revhead: " << ss4.str() << " | "<< 
          rpy[0]*180/M_PI << " " << rpy[1]*180/M_PI << " " << rpy[2]*180/M_PI << "\n";        
      }
      estimator_->setBodyPose(revised_local_to_head);

      // Publish the Position of the floating head:
      estimator_->publishUpdate(utime, revised_local_to_head, cl_cfg_.output_signal, false);



    }
    if (imu_counter_ > cl_cfg_.correction_frequency) { imu_counter_ =0; }
    imu_counter_++;
  }
}


Eigen::Isometry3d KDLToEigen(KDL::Frame tf){
  Eigen::Isometry3d tf_out;
  tf_out.setIdentity();
  tf_out.translation()  << tf.p[0], tf.p[1], tf.p[2];
  Eigen::Quaterniond q;
  tf.M.GetQuaternion( q.x() , q.y(), q.z(), q.w());
  tf_out.rotate(q);
  return tf_out;
}



int temp_counter = 0;
void StereoOdom::microstrainHandler(const lcm::ReceiveBuffer* rbuf, 
     const std::string& channel, const  bot_core::ins_t* msg){
  temp_counter++;
  if (temp_counter > 5){
    //std::cout << msg->gyro[0] << ", " << msg->gyro[1] << ", " << msg->gyro[2] << " rotation rate, gyro frame\n";
    temp_counter=0;
  }

  Eigen::Quaterniond local_to_body_orientation_from_imu;
  local_to_body_orientation_from_imu = imuOrientationToRobotOrientation(msg);
  fuseInterial(local_to_body_orientation_from_imu, msg->utime);

  // Transform rotation Rates into body frame:
  double camera_ang_vel_from_imu_[3];
  Eigen::Quaterniond imu_to_camera_quat = Eigen::Quaterniond( imu_to_camera_.rotation() );
  double imu_to_camera_quat_array[4];
  imu_to_camera_quat_array[0] =imu_to_camera_quat.w(); imu_to_camera_quat_array[1] =imu_to_camera_quat.x();
  imu_to_camera_quat_array[2] =imu_to_camera_quat.y(); imu_to_camera_quat_array[3] =imu_to_camera_quat.z();
  bot_quat_rotate_to( imu_to_camera_quat_array, msg->gyro, camera_ang_vel_from_imu_);
  camera_angular_velocity_from_imu_ = Eigen::Vector3d(camera_ang_vel_from_imu_[0], camera_ang_vel_from_imu_[1], camera_ang_vel_from_imu_[2]);
  // Didn't find this necessary - for smooth motion
  //camera_angular_velocity_from_imu_alpha_ = 0.8*camera_angular_velocity_from_imu_alpha_ + 0.2*camera_angular_velocity_from_imu_;

  camera_linear_velocity_from_imu_  = Eigen::Vector3d(0,0,0);

  // experimentally correct for sensor timing offset:
  int64_t temp_utime = msg->utime;// + 120000;
  estimator_->publishPose(temp_utime, "POSE_IMU_RATES", Eigen::Isometry3d::Identity(), camera_linear_velocity_from_imu_, camera_angular_velocity_from_imu_);
  // estimator_->publishPose(temp_utime, "POSE_IMU_RATES", Eigen::Isometry3d::Identity(), camera_linear_velocity_from_imu_, camera_angular_velocity_from_imu_alpha_);
}


int main(int argc, char **argv){
  CommandLineConfig cl_cfg;
  cl_cfg.camera_config = "CAMERA";
  cl_cfg.input_channel = "CAMERA";
  cl_cfg.output_signal = "POSE_BODY";
  cl_cfg.feature_analysis = FALSE; 
  cl_cfg.fusion_mode = 0;
  cl_cfg.verbose = false;
  cl_cfg.output_extension = "";
  cl_cfg.correction_frequency = 1;//; was typicall unused at 100;
  cl_cfg.atlas_version = 5;
  cl_cfg.feature_analysis_publish_period = 1; // 5
  cl_cfg.imu_channel = "IMU_MICROSTRAIN";
  std::string param_file = ""; // actual file
  cl_cfg.param_file = ""; // full path to file

  ConciseArgs parser(argc, argv, "simple-fusion");
  parser.add(cl_cfg.camera_config, "c", "camera_config", "Camera Config block to use: CAMERA, stereo, stereo_with_letterbox");
  parser.add(cl_cfg.output_signal, "p", "output_signal", "Output POSE_BODY and POSE_BODY_ALT signals");
  parser.add(cl_cfg.feature_analysis, "f", "feature_analysis", "Publish Feature Analysis Data");
  parser.add(cl_cfg.feature_analysis_publish_period, "fp", "feature_analysis_publish_period", "Publish features with this period");  
  parser.add(cl_cfg.fusion_mode, "m", "fusion_mode", "0 none, 1 at init, 2 rpy, 3 rp only, (both continuous)");
  parser.add(cl_cfg.input_channel, "i", "input_channel", "input_channel - CAMERA or CAMERA_BLACKENED");
  parser.add(cl_cfg.output_extension, "o", "output_extension", "Extension to pose channels (e.g. '_VO' ");
  parser.add(cl_cfg.correction_frequency, "y", "correction_frequency", "Correct the R/P every XX IMU measurements");
  parser.add(cl_cfg.verbose, "v", "verbose", "Verbose printf");
  parser.add(cl_cfg.imu_channel, "imu", "imu_channel", "IMU channel to listen for");
  parser.add(cl_cfg.atlas_version, "a", "atlas_version", "Atlas version to use");
  parser.add(cl_cfg.in_log_fname, "L", "in_log_fname", "Process this log file");
  parser.add(param_file, "P", "param_file", "Pull params from this file instead of LCM");
  parser.parse();
  cout << cl_cfg.fusion_mode << " is fusion_mode\n";
  cout << cl_cfg.camera_config << " is camera_config\n";
  
  cl_cfg.param_file = std::string(getConfigPath()) +'/' + std::string(param_file);
  if (param_file.empty()) { // get param from lcm
    cl_cfg.param_file = "";
  }

  //
  bool running_from_log = !cl_cfg.in_log_fname.empty();
  boost::shared_ptr<lcm::LCM> lcm_recv;
  boost::shared_ptr<lcm::LCM> lcm_pub;
  if (running_from_log) {
    printf("running from log file: %s\n", cl_cfg.in_log_fname.c_str());
    //std::string lcmurl = "file://" + in_log_fname + "?speed=0";
    std::stringstream lcmurl;
    //lcmurl << "file://" << in_log_fname << "?speed=" << processing_rate << "&start_timestamp=" << begin_timestamp;
    lcmurl << "file://" << cl_cfg.in_log_fname ;
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

  StereoOdom fo= StereoOdom(lcm_recv, lcm_pub, cl_cfg);
  while(0 == lcm_recv->handle());
}
