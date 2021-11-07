#include <list>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>



#include "graph.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "batchdata.h"
//#include "iris.h"
#include "mnist.h"
#include "optimizer_adam.h"
#include "optimizer_sgd_moment.h"
//#include "word_embed.h"

using namespace std;

MallocCounter mallocCounter;


void asMatrix(PVariable x1, float *X){
    x1->data.memSetHost(X);
}


float getAccurecy(Graph *g_softmax, PVariable h, PVariable d, int batchSize){
    PVariable y = ((Softmax *)g_softmax)->forward(h);

    int *maxIdx_z3 = new int(batchSize);
    y->data.maxRowIndex(maxIdx_z3);

    int *maxIdx_d = new int(batchSize);
    d->data.maxRowIndex(maxIdx_d);

    int hit = 0;
    for(int i=0; i<batchSize; i++){
        if (maxIdx_d[i] == maxIdx_z3[i]) hit++;
    }
    float accurecy = ((float)hit) / ((float) batchSize);
    delete maxIdx_d;
    delete maxIdx_z3;
    return accurecy;
}

int main(){


    int epochNums = 30;
    int epochAENums = 20;
    int totalSampleSize = 60000;
    int totalTestSize = 10000;

    int batchSize = 100;
    int i_size = 784;
    int n_size = 1024;
    int o_size = 10;
    float learning_rate = 0.001;
    float ae_learning_rate = 0.001;
    float dropout_p = 0.5;
    float ae_dropout_p = 0.5;

    cout << "init dataset..." << endl;
    vector<vector<float>> train_data, test_data;
    vector<float> label_data, label_test_data;

    Mnist mnist, mnist_test;
    train_data = mnist.readTrainingFile("train-images-idx3-ubyte");
    label_data = mnist.readLabelFile("train-labels-idx1-ubyte");
    test_data = mnist_test.readTrainingFile("t10k-images-idx3-ubyte");
    label_test_data = mnist_test.readLabelFile("t10k-labels-idx1-ubyte");

    Dataset *dataset = new Dataset();
    dataset->standrize(&train_data);
    vector<BatchData *> bds;
    for(int i=0; i<totalSampleSize/batchSize; i++){
        BatchData *bdata = new BatchData(i_size, o_size, batchSize);
        dataset->createMiniBatch(train_data, label_data, bdata->getX(), bdata->getD(), batchSize, o_size, i);
        bds.push_back(bdata);
    }
    dataset->standrize(&test_data);
    vector<BatchData *> bds_test;
    for(int i=0; i<totalTestSize/batchSize; i++){
        BatchData *bdata = new BatchData(i_size, o_size, batchSize);
        dataset->createMiniBatch(test_data, label_test_data, bdata->getX(), bdata->getD(), batchSize, o_size, i);
        bds_test.push_back(bdata);
    }


    std::chrono::system_clock::time_point  start, end;

    cout << "create model..." << endl;
    Model model;
    model.putG("g1", new Linear(n_size, i_size));
    model.putG("g_relu1", new ReLU());
    model.putG("g_drop1", new Dropout(dropout_p));
    model.putG("g2", new Linear(n_size, n_size));
    model.putG("g_relu2", new ReLU());
    model.putG("g_drop2", new Dropout(dropout_p));
    model.putG("g3", new Linear(o_size, n_size));
    model.putG("g_softmax_cross_entoropy", new SoftmaxCrossEntropy());
    model.putG("g_softmax", new Softmax());

    OptimizerAdam optimizer(&model, learning_rate);
    optimizer.init();

    cout << "start training ..." << endl;
    for(int k=0; k<epochNums; k++){

        start = std::chrono::system_clock::now();

        std::random_shuffle(bds.begin(), bds.end());

        float sum_loss = 0.0;
        float accurecy = 0.0;

        PVariable loss_graph(new Variable(1, 1));

        for(int i=0; i<totalSampleSize/batchSize; i++){

            PVariable x1(new Variable(i_size, batchSize));
            PVariable d(new Variable(o_size, batchSize));

            // create mini-batch =========================
            float *X = bds.at(i)->getX();
            float *D = bds.at(i)->getD();
            asMatrix(x1, X);
            asMatrix(d, D);

            // forward ------------------------------------------
            PVariable h1 = model.G("g_relu1")->forward(model.G("g1")->forward(x1));
            PVariable h2 = model.G("g_relu2")->forward(model.G("g2")->forward(h1));

            PVariable h3 = model.G("g3")->forward(h2);
            PVariable loss = model.G("g_softmax_cross_entoropy")->forward(h3, d);

            // loss ---------------------------------------------
            sum_loss += loss->val();

            // backward -----------------------------------------
            loss->backward();

            // update -------------------------------------------
            optimizer.update();

            model.unchain();
        }


        end = std::chrono::system_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        float loss_mean = sum_loss/((float)totalSampleSize/batchSize);
        float accurecy_mean = accurecy/((float)totalSampleSize/batchSize);
        cout << "epoch:" << k+1 << " loss:" << loss_mean << " accurecy:" << setprecision(3) << accurecy_mean*100 << "% time:" << elapsed << "ms" << endl;
    }


    //cout << "saving model..." << endl;
    //model.save("mlp_test.model");

    //cout << "loading model..." << endl;
    //Model model_train;
    //model_train.load("mlp_test.model");
    //cout << "loaded" << endl;

    cout << "start predict..." << endl;
    float accurecy = 0.0;
    int predict_epoch = totalTestSize/batchSize;
    for(int i=0; i<predict_epoch; i++){

        std::random_shuffle(bds_test.begin(), bds_test.end());

        PVariable x1(new Variable(i_size, batchSize));
        PVariable d(new Variable(o_size, batchSize));

        // create mini-batch =========================
        float *X = bds_test.at(i)->getX();
        float *D = bds_test.at(i)->getD();
        asMatrix(x1, X);
        asMatrix(d, D);

        // forward ------------------------------------------
        PVariable h1 = model.G("g_relu1")->forward(model.G("g1")->forward(x1));
        PVariable h2 = model.G("g_relu2")->forward(model.G("g2")->forward(h1));

        PVariable h3 = model.G("g3")->forward(h2);

        accurecy += getAccurecy(model.G("g_softmax"), h3, d, batchSize);

        model.unchain();

    }
    cout << "accurecy: " << setprecision(3) << accurecy/((float)predict_epoch)*100 << "%" << endl;


}

