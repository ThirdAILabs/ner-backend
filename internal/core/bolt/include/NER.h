#pragma once

#include <memory>
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
  std::vector<std::vector<LabeledEntity>>
  predict(const std::vector<std::string> &sentences);

  void train(const std::string &filename, float learning_rate, uint32_t epochs);

  static std::unique_ptr<NerModel> load(const std::string &path);
};

} // namespace thirdai::automl::udt