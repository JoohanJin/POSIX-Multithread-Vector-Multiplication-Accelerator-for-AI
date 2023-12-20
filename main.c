/************************************************************
* Filename: llama2_3035661360.c
* Student name and Number: JIN, Joohan 3035661360
* Development platform: WSL Ubuntu 22.04, gcc version 11.4.0
* Remark:
*
*************************************************************/

/*
PLEASE WRITE DOWN NAME AND UID BELOW BEFORE SUBMISSION
* NAME:
* UID :

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o template template.c utilities.c -O2 -pthread -lm
gcc -o llama2_3035661360 llama2_3035661360.c utilities.c -O2 -pthread -lm

Then Run with:
$ ./parallel
./llama2_3035661360 <seed> <thr_count>
*/

#define _GNU_SOURCE // keep this line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-vector multiplication, used in QKV Mapping and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each other, so we can use parallel computing for acceleration.
 * 
 * Please use <pthread.h> and your favorite control method,
 * semaphore (please #include <semaphore.h>) / mutex lock + conditional variable
 * 
 * A sequential version is provided below, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// additional header file
// to make a use of a binary semaphore for editting the out
#include <pthread.h>
#include <semaphore.h>

// global variables
typedef struct thread_data{
    float *out;
    float *mat;
    float *vec;
    int col; 
    int row;
} param_array;

struct rusage main_usage;
struct rusage *thread_usage;

int num_of_threads;
pthread_t *threads;
param_array *parameters;
sem_t outSem;
sem_t mSleepSem;
sem_t countSem;
sem_t rscSem;
sem_t* tSleepSem;
int tnum_flag = 0;
int thread_executing = 1; // boolean value to check if the program is done or not

void *thr_func(void *arg) {
    int id = *((int *)arg);

    sem_wait(&rscSem);
    param_array *pa = &(parameters[id]);
    sem_post(&rscSem);

    while (1){
        sem_wait(&tSleepSem[id]);

        if (!thread_executing){
            break;
        }

        // Implement calculation

        sem_wait(&rscSem);
        int row = pa->row;
        int col = pa->col;
        float *vec = pa->vec;
        float *mat = pa->mat;
        sem_post(&rscSem);

        // int start = id * (int)((float)row/num_of_threads);
        // int end = (id + 1) * (int)((float)row/num_of_threads) - 1;

        int size = (int)((float)row/num_of_threads);
        if (row % num_of_threads == 0) {
            size = row / num_of_threads;
        }
        else {
            size = (row / num_of_threads) + 1;
        }

        int start = size * id;
        int end;
        if (id == num_of_threads - 1){
            end = row - 1;
        }
        else{
            end = start + size - 1;
        }


        // printf("id: %d, start: %d, end: %d, row: %d\n", id, start, end, row);

        for (int i = start; i <= end; i++){
            float val = 0.0f;

            if (i < row){
                for (int j = 0; j < col; j++){
                    val += mat[i * col + j] * vec[j];
                }
                sem_wait(&outSem);
                pa->out[i] = val;
                sem_post(&outSem);
            }
        }


        // decrement the int glag showing the number of thread after its execution
        sem_post(&mSleepSem);
    }
    // need to get the usage data and upload them on usage array.
    getrusage(RUSAGE_THREAD, &thread_usage[id]);
    // exit the calling thread
    free(arg);
    // printf("yeah in thread\n");

}

int init_mat_vec_mul(int thr_count) {
    num_of_threads = thr_count;

    sem_init(&outSem, 0, 1);
    sem_init(&mSleepSem, 0, 0);
    sem_init(&countSem, 0, 1);
    sem_init(&rscSem, 0, 1);

    tSleepSem = (sem_t*)malloc(sizeof(sem_t) * num_of_threads);
    for (int i = 0; i < num_of_threads; i++){
        sem_init(&tSleepSem[i], 0, 0);
    }

    threads = (pthread_t*)malloc(sizeof(pthread_t) * num_of_threads);
    parameters = (param_array*)malloc(sizeof(param_array) * num_of_threads);
    thread_usage = (struct rusage*)malloc(sizeof(struct rusage) * num_of_threads);

    for (int i = 0; i < thr_count; i++){
        int *id = (int *) malloc (sizeof(int));
        *id = i;
        pthread_create(&threads[i], NULL, thr_func, (void *) id);
    }

    return 0;
}


void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    // before waking up threads, we need to upload global variable so that each thread can access on it
    sem_wait(&rscSem);
    for (int i = 0; i < num_of_threads; i++){
        parameters[i].out = out;
        parameters[i].mat = mat;
        parameters[i].vec = vec;
        parameters[i].col = col;
        parameters[i].row = row;
    }
    sem_post(&rscSem);
    
    // waking up each thread
    for (int i = 0; i < num_of_threads; i++){
        sem_post(&tSleepSem[i]);
    }
    // wake up and wait
    for (int i = 0; i < num_of_threads; i++){
        sem_wait(&mSleepSem);
    }
}


int close_mat_vec_mul() {
    thread_executing = 0;
    for (int i = 0; i < num_of_threads; i++){
        sem_post(&tSleepSem[i]);
    }

    for (int i = 0; i < num_of_threads; i++){
        pthread_join(threads[i], NULL);
    }

    getrusage(RUSAGE_SELF, &main_usage);
    
    for (int i = 0; i < num_of_threads; i++){
        float user_time = thread_usage[i].ru_utime.tv_sec + thread_usage[i].ru_utime.tv_usec / 1000000;
        float system_time = thread_usage[i].ru_stime.tv_sec + thread_usage[i].ru_stime.tv_usec / 1000000;
        printf("Thread %d has complted - user: %f s, system: %f s\n", i, user_time, system_time);
    }
    float main_user_time = main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec / 1000000;
    float main_system_time = main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_sec / 1000000;
    printf("main thread - user: %f s, system: %f s\n", main_user_time, main_system_time);

    // free memory allocated
    free(threads);
    free(parameters);
    // free(thread_usage);

    for (int i = 0; i < num_of_threads; i++){
        sem_destroy(&tSleepSem[i]);
    }
    free(tSleepSem);

    sem_destroy(&outSem);
    sem_destroy(&mSleepSem);
    sem_destroy(&countSem);
    sem_destroy(&rscSem);
}
// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim = p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}