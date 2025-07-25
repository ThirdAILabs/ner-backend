#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StringList_t StringList_t;
StringList_t *StringList_new();
void StringList_free(StringList_t *self);
void StringList_append(StringList_t *self, const char *value);

typedef struct Results_t Results_t;

void Results_free(Results_t *self);

unsigned int Results_batch_size(const Results_t *self);

unsigned int Results_len(const Results_t *self, unsigned int batch_index);

const char *Results_label(const Results_t *self, unsigned int batch_index,
                          unsigned int index);

const char *Results_text(const Results_t *self, unsigned int batch_index,
                         unsigned int index);

unsigned int Results_start(const Results_t *self, unsigned int batch_index,
                           unsigned int index);

unsigned int Results_end(const Results_t *self, unsigned int batch_index,
                         unsigned int index);

typedef struct NER_t NER_t;

NER_t *NER_load(const char *path, const char **err_ptr);

void NER_free(NER_t *self);

Results_t *NER_predict(const NER_t *self, const StringList_t *sentences,
                       const char **err_ptr);

void NER_train(const NER_t *self, const char *filename, float learning_rate,
               unsigned int epochs, const char **err_ptr);


void NER_source_target_cols(const NER_t *self, char **tokens_col, char **tags_col);

void NER_source_target_cols_free(char *tokens_col, char *tags_col);

void NER_save(const NER_t *self, const char *path, const char **err_ptr);

#ifdef __cplusplus
}
#endif