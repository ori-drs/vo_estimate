#include <isam/isam.h>
#include <iostream>
#include <vector>

#include <lcmtypes/vs_object_collection_t.h>
#include <lcmtypes/vs_point3d_list_collection_t.h>
#include <bot_core/bot_core.h>
#include <pronto_utils/pronto_math.hpp>

using namespace std;
using namespace isam;
using namespace Eigen;

lcm_t* m_lcm;

void sendCollection(vector < Pose2d_Node* > nodes,int id,string label)
{
  vs_object_collection_t objs;  
  size_t n = nodes.size();
  if (n > 1) {
    objs.id = id;
    stringstream oss;
    string mystr;
    oss << "coll" << id;
    mystr=oss.str();    

    objs.name = (char*) label.c_str();
    objs.type = VS_OBJECT_COLLECTION_T_POSE3D;
    objs.reset = false;
    objs.nobjects = n;
    vs_object_t poses[n];
    for (size_t i = 0; i < n; i++) {
      Pose2d_Node* node =  nodes[i];
      isam::Pose2d  pose;
      pose = (*node).value();

      // direct Eigen Approach: TODO: test and use
      //Matrix3f m;
      //m = AngleAxisf(angle1, Vector3f::UnitZ())
      // *  * AngleAxisf(angle2, Vector3f::UnitY())
      // *  * AngleAxisf(angle3, Vector3f::UnitZ());
      Eigen::Quaterniond quat = euler_to_quat(0,0, pose.t());

      poses[i].id = i;
      poses[i].x = pose.x();
      poses[i].y = pose.y() ;
      poses[i].z = 0;
      poses[i].qw = quat.w();
      poses[i].qx = quat.x();
      poses[i].qy = quat.y();
      poses[i].qz = quat.z();
    }
    objs.objects = poses;
    vs_object_collection_t_publish(m_lcm, "OBJECT_COLLECTION", &objs);
  }
}


int main() {
  m_lcm = lcm_create(NULL);
  Slam slam;
  MatrixXd sqrtinf = 0.5 * Matrix<double, 3, 3>::Identity();
  sqrtinf(2,2) = 1;
  MatrixXd prior_noise = 0.0001 * Matrix<double, 3, 3>::Identity();



  /////////////
  Pose2d prior_origin(0., 0., 0.);
  Pose2d_Node a0;
  slam.add_node(&a0);
  Pose2d_Factor p_a0(&a0, prior_origin, Covariance(prior_noise));
  slam.add_factor(&p_a0);
  vector < Pose2d_Node* > pg1_nodes;
  pg1_nodes.push_back(&a0);

  /////////////
  Pose2d odo(1, 0., 0.); // A
  Pose2d_Node a1;
  slam.add_node(&a1);
  pg1_nodes.push_back(&a1);
  Pose2d_Pose2d_Factor o_a01(&a0, &a1, odo, Covariance(sqrtinf));
  slam.add_factor(&o_a01);

  /////////////
  Pose2d odo3(1, 0., 0.); // B
  Pose2d_Node a3;
  slam.add_node(&a3);
  pg1_nodes.push_back(&a3);
  Pose2d_Pose2d_Factor o_a23(&a1, &a3, odo3, Covariance(sqrtinf));
  slam.add_factor(&o_a23);  
  
  /////////////
  Pose2d odo4(1, 0., 0.); // C
  Pose2d_Node a4;
  slam.add_node(&a4);
  pg1_nodes.push_back(&a4);
  Pose2d_Pose2d_Factor o_a34(&a3, &a4, odo4, Covariance(sqrtinf));
  slam.add_factor(&o_a34);    
  
  /////////////
  Pose2d odo5(1, 0., 0.); // D
  Pose2d_Node a5;
  slam.add_node(&a5);
  pg1_nodes.push_back(&a5);
  Pose2d_Pose2d_Factor o_a45(&a4, &a5, odo5, Covariance(sqrtinf));
  slam.add_factor(&o_a45);  
  // sendCollection(pg1_nodes,10,"Part map"); // as an example

  Pose2d odo6(1, 0., 0.); // E
  Pose2d_Node a6;
  slam.add_node(&a6);
  pg1_nodes.push_back(&a6);
  Pose2d_Pose2d_Factor o_a56(&a5, &a6, odo6, Covariance(sqrtinf));
  slam.add_factor(&o_a56);

  // 3 Publish pose graphs without and loop
  slam.update();

  /////////////
  Pose2d factor_origin(5., 2., 0.);
  Pose2d_Node factor_node;
  slam.add_node(&factor_node);
  Pose2d_Factor p_factor(&factor_node, factor_origin, Covariance(prior_noise));
  slam.add_factor(&p_factor);
  pg1_nodes.push_back(&factor_node);

  slam.update();
  sendCollection(pg1_nodes,11,"Map - no loop");

  ///////////// Update the last pose to get closed to (5,2)
  Pose2d small_pos(0.,0.,0.);
  Pose2d_Pose2d_Factor o_a6tofactor(&factor_node, &a6, small_pos, Covariance(sqrtinf));
  slam.add_factor(&o_a6tofactor);

  // 6. Optimize and publish to lcm:
  slam.update();
  slam.batch_optimization();
  std::cout << "\n";
  
  sendCollection(pg1_nodes,12,"Map - loop, optimized");

  std::cout << "Exiting simple_isam__factor_demo\n";

  return 0;
}
