// add new node
// retrieve all nodes (keep a vector of nodes?)
// retrieve translation for a node
// retrieve rotation for a node
// 

#include <isam/isam.h>

struct Pose {
  double x;
  double y;
  double t;
};

struct Node {
  Pose2d_Node isamNode;
};

class IsamWrapper {
public:
	IsamWrapper();
  ~IsamWrapper();

  getNodes() { return nodes_};

  std::vector<double> getTranslation(const Node node) {
    std::vector<double> trans; 
    trans.push_back(node.isamNode.value().x());
    trans.push_back(node.isamNode.value().y());
    return trans;
  };

  double getRotation(const Node node) { return node.isamNode.value().t();};

  void update() { slam_.update(); };

  void optimize() { slam_.batch_optimization(); };

  void addNode() {};

private:
  Slam slam_;
  std::vector<Node*> nodes_;
};