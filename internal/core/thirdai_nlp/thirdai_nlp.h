#pragma once

#include <stddef.h>

struct Prediction {
  size_t start;
  size_t end;
  char *label;
};

struct Predictions {
  struct Prediction *predictions;
  size_t *batch_offsets;
  size_t batch_size;
  char *error_msg;
};

struct LoadResult {
  void *model_ptr;
  char *error_msg;
};

struct LoadResult cnn_model_load(const char *model_path);

void cnn_model_free_load_result(struct LoadResult load_result);

void cnn_model_free(void *model_ptr);

struct Predictions cnn_model_predict(void *model_ptr, const char **batch,
                                     size_t batch_size);

void free_predictions(struct Predictions predictions);