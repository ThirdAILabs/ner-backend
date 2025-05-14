#include "binding.h"
#include "NER.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using thirdai::automl::udt::LabeledEntity;
using thirdai::automl::udt::NerModel;

struct StringList_t {
  std::vector<std::string> list;
};

StringList_t *StringList_new() { return new StringList_t(); }

void StringList_free(StringList_t *self) { delete self; }

void StringList_append(StringList_t *self, const char *value) {
  self->list.emplace_back(value);
}

struct Results_t {
  std::vector<std::vector<LabeledEntity>> results;
};

void Results_free(Results_t *self) { delete self; }

unsigned int Results_batch_size(const Results_t *self) {
  return self->results.size();
}

unsigned int Results_len(const Results_t *self, unsigned int batch_index) {
  return self->results.at(batch_index).size();
}

const char *Results_label(const Results_t *self, unsigned int batch_index,
                          unsigned int index) {
  return self->results.at(batch_index).at(index).label.c_str();
}

const char *Results_text(const Results_t *self, unsigned int batch_index,
                         unsigned int index) {
  return self->results.at(batch_index).at(index).text.c_str();
}

unsigned int Results_start(const Results_t *self, unsigned int batch_index,
                           unsigned int index) {
  return self->results.at(batch_index).at(index).start;
}

unsigned int Results_end(const Results_t *self, unsigned int batch_index,
                         unsigned int index) {
  return self->results.at(batch_index).at(index).end;
}

void copyError(const std::exception &e, const char **err_ptr) {
  char *err_msg = new char[std::strlen(e.what()) + 1];
  std::strcpy(err_msg, e.what());
  *err_ptr = err_msg;
}

struct NER_t {
  std::unique_ptr<NerModel> ner;
};

NER_t *NER_load(const char *path, const char **err_ptr) {
  try {
    auto model = NerModel::load(path);
    auto *ner = new NER_t();
    ner->ner = std::move(model);
    return ner;
  } catch (const std::exception &e) {
    copyError(e, err_ptr);
    return nullptr;
  }
}

void NER_free(NER_t *self) { delete self; }

Results_t *NER_predict(const NER_t *self, const StringList_t *sentences,
                       const char **err_ptr) {
  try {
    auto results = self->ner->predict(sentences->list);
    auto *results_ptr = new Results_t();
    results_ptr->results = std::move(results);
    return results_ptr;
  } catch (const std::exception &e) {
    copyError(e, err_ptr);
    return nullptr;
  }
}

void NER_train(const NER_t *self, const char *filename, float learning_rate,
               unsigned int epochs, const char **err_ptr) {
  try {
    self->ner->train(filename, learning_rate, epochs);
  } catch (const std::exception &e) {
    copyError(e, err_ptr);
  }
}

// get CSV column names: tokens and tags
// since C does not have string support, we use char* to store the column names
void NER_source_target_cols(const NER_t *self, const char **tokens_col, const char **tags_col) {
  auto cols = self->ner->sourceTargetCols();
  const std::string &tokens_str = cols.first;
  const std::string &tags_str = cols.second;
  size_t tok_len = tokens_str.size() + 1;
  char *c_tokens = new char[tok_len];
  std::strcpy(c_tokens, tokens_str.c_str());
  *tokens_col = c_tokens;
  size_t tag_len = tags_str.size() + 1;
  char *c_tags = new char[tag_len];
  std::strcpy(c_tags, tags_str.c_str());
  *tags_col = c_tags;
}

// free memory allocated by NER_source_target_cols
void NER_source_target_cols_free(const char *tokens_col, const char *tags_col) {
  delete [] tokens_col;
  delete [] tags_col;
}

void NER_save(const NER_t *self, const char *path, const char **err_ptr) {
  try {
    self->ner->save(path);
  } catch (const std::exception &e) {
    copyError(e, err_ptr);
  }
}