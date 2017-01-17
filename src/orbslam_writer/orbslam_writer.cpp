#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>      // std::ifstream, std::ofstream

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <boost/shared_ptr.hpp>
#include <lcm/lcm-cpp.hpp>

#include "lcmtypes/bot_core.hpp"
#include <image_io_utils/image_io_utils.hpp> // to simplify jpeg/zlib compression and decompression
#include <ConciseArgs>
using namespace cv;
using namespace std;

class Pass{
  public:
    Pass(boost::shared_ptr<lcm::LCM> &lcm_, bool verbose_,
         std::string input_channel_, std::string output_folder_);
    
    ~Pass(){
      std::cout << "finish\n";
      timestamp_file_.close();

    }    
  private:
    boost::shared_ptr<lcm::LCM> lcm_;
    bool verbose_;
    std::string input_channel_;
    std::string output_folder_;
    image_io_utils*  imgutils_;
    uint8_t* img_buf_; 
    uint8_t* rgb_compress_buffer_;

    ofstream timestamp_file_;

    
    void imagesHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::images_t* msg);
};

Pass::Pass(boost::shared_ptr<lcm::LCM> &lcm_, bool verbose_,
         std::string input_channel_, std::string output_folder_):
    lcm_(lcm_), verbose_(verbose_), 
    input_channel_(input_channel_), output_folder_(output_folder_){
  lcm_->subscribe( input_channel_ ,&Pass::imagesHandler,this);

  // left these numbers very large:
  img_buf_= (uint8_t*) malloc(3* 1524  * 1544);
  imgutils_ = new image_io_utils( lcm_, 
                                  1524, 
                                  3*1544 );  

  rgb_compress_buffer_= (uint8_t*) malloc(3* 1524  * 1544);

  std::stringstream ss;
  ss << output_folder_ << "timestamps.txt";

  timestamp_file_.open (ss.str().c_str() );


}


void Pass::imagesHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::images_t* msg){
  std::cout << "1got " << channel << "\n";

  int w = msg->images[0].width;
  int h = msg->images[0].height;
  int n_colors =3;
  
  imgutils_->decodeImageToGray( & msg->images[0], img_buf_);
  cv::Mat img(cv::Size( w, h),CV_8UC1);
  img.data = img_buf_;
  std::stringstream ss;
  ss << output_folder_ << "left_cam/" << msg->utime << ".png";
  std::cout << ss.str() << "\n";
  //cv::cvtColor(img, img, CV_RGB2BGR);
  imwrite( ss.str(), img);



  imgutils_->decodeImageToGray( & msg->images[1], img_buf_);
  cv::Mat img2(cv::Size( w, h),CV_8UC1);
  img2.data = img_buf_;
  std::stringstream ss2;
  ss2 << output_folder_ << "right_cam/" << msg->utime << ".png";
  std::cout << ss2.str() << "\n";
  //cv::cvtColor(img, img, CV_RGB2BGR);
  imwrite( ss2.str(), img2);



  timestamp_file_ << msg->utime << "\n";
  timestamp_file_.flush();

}


int main( int argc, char** argv ){
  ConciseArgs parser(argc, argv, "blacken-image");
  bool verbose=false;
  string input_channel="MULTISENSE_CAMERA";
  string output_folder="/media/mfallon/bay_drive/hyq_stereo/";
  parser.add(verbose, "v", "verbose", "Verbosity");
  parser.add(input_channel, "l", "input_channel", "Incoming channel");
  parser.add(output_folder, "o", "output_folder", "Output folder");
  parser.parse();
  cout << verbose << " is verbose\n";
  cout << input_channel << " is input_channel\n";
  cout << output_folder << " is output_folder\n";
  
  boost::shared_ptr<lcm::LCM> lcm(new lcm::LCM);
  if(!lcm->good()){
    std::cerr <<"ERROR: lcm is not good()" <<std::endl;
  }


  
  Pass app(lcm,verbose,input_channel, output_folder);
  cout << "Ready to convert from imu to pose" << endl << "============================" << endl;
  while(0 == lcm->handle());

  std::cout << "exit\n";
  return 0;
}
