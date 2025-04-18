#pragma once

#include <string>
#include <vector>

namespace thirdai::automl::udt {

struct LabeledEntity {
  std::string label;
  std::string text;
  size_t start;
  size_t end;
};

class NerModel {
public:
  // std::unordered_map<std::string, std::vector<float>> train(
  //     const dataset::DataSourcePtr& data, float learning_rate, uint32_t
  //     epochs, const std::vector<std::string>& train_metrics, const
  //     dataset::DataSourcePtr& val_data, const std::vector<std::string>&
  //     val_metrics, const std::vector<bolt::callbacks::CallbackPtr>&
  //     callbacks, TrainOptions options, const bolt::DistributedCommPtr& comm);

  // std::unordered_map<std::string, std::vector<float>> evaluate(
  //     const dataset::DataSourcePtr& data,
  //     const std::vector<std::string>& metrics, bool sparse_inference,
  //     bool verbose);

  std::vector<std::vector<LabeledEntity>>
  predict(const std::vector<std::string> &sentences);

  static std::unique_ptr<NerModel> load(const std::string &path);
};

} // namespace thirdai::automl::udt