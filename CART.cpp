#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "stdlib.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>

struct split_node
{
    int bestFeatureIndex;
    double bestFeatureValue;
};


struct CART
{
    bool is_leaf=true;
    int bestSplitFeature=0;
    double bestSplitValue=0;
    struct CART* left_tree;
    struct CART* right_tree;
    double value=0;
};


//Split the matrix into matLeft and matRight, based on the featureIndex and featureValue given
std::vector<std::vector<std::vector<double> > > binarySplit(std::vector<std::vector<double> > dataset,int feature,double value){
	std::vector<std::vector<double> > matLeft,matRight;
	std::vector<std::vector<std::vector<double> > > res;
	for ( int i=0 ; i<dataset.size() ; i++ ){
		if(dataset[i][feature] <= value) {matLeft.push_back(dataset[i]);}
		else {matRight.push_back(dataset[i]);}
		}
	res.push_back(matLeft);
	res.push_back(matRight);
	return res;
	}


//Calculate the average value of the leaf
double regressLeaf(std::vector<std::vector<double> > dataset){
	double tot = 0;
    if(dataset.size() == 0) return 0;
	for ( int i=0 ; i<dataset.size() ; i++ ){tot += dataset[i][dataset[0].size()-1];}
	return tot/dataset.size();
	}


//Customized compare function for vector sorting
struct compare{
    int index;
    bool operator()(std::vector<double>  v1, std::vector<double> v2){
        return v1[index] < v2[index];
    }
};

//Find the best split value given the featureIndex
std::vector<double> getError(std::vector<std::vector<double> > dataset,int feature, int thresholdSamples){
    compare cmp;
    cmp.index = feature;
    std::sort(dataset.begin(), dataset.end(),cmp);
    double tot_sum = 0, tot_sum_square = 0, variance;
	for ( int i=0 ; i<dataset.size() ; i++ ){
		tot_sum += dataset[i][dataset[0].size()-1]; 
		tot_sum_square += dataset[i][dataset[0].size()-1]* dataset[i][dataset[0].size()-1];
		}
	double cur = 0, tmpVariance = INFINITY;
	int tmpIndex = thresholdSamples;
	for( int i = 0; i < dataset.size()-thresholdSamples; i++ ){
		cur += dataset[i][dataset[0].size()-1];
		variance = -cur*cur/(i+1)-(tot_sum-cur)*(tot_sum-cur)/(dataset.size()-i-1);
        if (variance < tmpVariance and i > thresholdSamples) {tmpVariance = variance; tmpIndex = i;} 
		}
	std::vector<double> res;
	res.push_back(tot_sum_square+tmpVariance);
	res.push_back(dataset[tmpIndex][feature]);
	return res;
	}


//Loop over all indexes, find which index reduces the error most
struct split_node chooseBestSplit(std::vector<std::vector<double> > dataset, int thresholdSamples){
    struct split_node res;
	if (dataset.size() <= thresholdSamples) {
        res.bestFeatureIndex = -1; 
        res.bestFeatureValue= regressLeaf(dataset); 
        return res;
        }
    int m = dataset.size();
    int n = dataset[0].size();
    double Err = INFINITY;
    double bestErr = INFINITY;
    int bestFeatureIndex = 0;
    double bestFeatureValue = 0;
    for( int i =0; i < n-1; i++){
        std::vector<double> getErr_res = getError(dataset,i,thresholdSamples);
        double tmpErr = getErr_res[0];
        double featureValue = getErr_res[1];
        if(tmpErr<bestErr){
            bestErr = tmpErr;
            bestFeatureIndex = i;
            bestFeatureValue = featureValue;
            }
        }
    res.bestFeatureIndex = bestFeatureIndex;
    res.bestFeatureValue = bestFeatureValue;
    return res;
	}


//Create a split node, and call itself recursively, until reaches max_depth
struct CART* createCART(std::vector<std::vector<double> > dataset, int thresholdSamples, int depth){
    struct CART* tree_node = new CART;
    if(depth == 0 or dataset.size() <= thresholdSamples){
        tree_node->is_leaf = true;
        tree_node->value = regressLeaf(dataset);
        return tree_node;    
    }
    struct split_node node = chooseBestSplit(dataset,thresholdSamples);
    int feature = node.bestFeatureIndex;
    double value = node.bestFeatureValue;
    if(feature == -1){
        tree_node->is_leaf = true;
        tree_node->value = value;
        return tree_node;
    }

    tree_node->is_leaf = false;
    tree_node->bestSplitFeature = feature;
    tree_node->bestSplitValue = value;
    std::vector<std::vector<std::vector<double> > > splited_set = binarySplit(dataset,feature,value);
    std::vector<std::vector<double> > left_set, right_set;
    left_set = splited_set[0];
    right_set = splited_set[1];

    tree_node->left_tree = createCART(left_set,thresholdSamples,depth-1);;
    tree_node->right_tree = createCART(right_set,thresholdSamples,depth-1);; 
    return tree_node;
}


//Evaluate the value of the regress tree
double valuePredict(struct CART tree_node, std::vector<double> inputData){
    if(tree_node.is_leaf) return tree_node.value;
    if(inputData[tree_node.bestSplitFeature] <= tree_node.bestSplitValue) return valuePredict(*tree_node.left_tree, inputData);
    else return valuePredict(*tree_node.right_tree, inputData);
}


//Evaluate the test dataset
std::vector<double> predict(struct CART tree_node, std::vector<std::vector<double> > testData){
    int m = testData.size();
    std::vector<double> yHat;
    for(int i = 0; i < m; i++) yHat.push_back(valuePredict(tree_node,testData[i]));
    return yHat;
}


//The tree is basically a function, here we calculate the integral of the function on the domain
double tree_evaluator(struct CART tree_node, std::vector<std::vector<double> > intervals){
    if(tree_node.is_leaf){
        double volumn = 1;
        for(int i = 0;i <intervals.size();i++) volumn *= (intervals[i][1] - intervals[i][0]);
        return tree_node.value * volumn;
    }
    std::vector<std::vector<double> > left_tree_intervals(intervals), right_tree_intervals(intervals);
    left_tree_intervals[tree_node.bestSplitFeature][1] = std::min(left_tree_intervals[tree_node.bestSplitFeature][1],tree_node.bestSplitValue);
    right_tree_intervals[tree_node.bestSplitFeature][0] = std::max(right_tree_intervals[tree_node.bestSplitFeature][0],tree_node.bestSplitValue);
    return tree_evaluator(*tree_node.left_tree, left_tree_intervals) + tree_evaluator(*tree_node.right_tree, right_tree_intervals);
}


//Additive tree, Boosted Decision Tree
std::vector<struct CART* > create_BDT(std::vector<std::vector<double> > dataset,int thresholdSamples = 4, int depth=2,int n_est = 10,double lr =0.1){
    std::vector<struct CART* > res;
    std::vector<std::vector<double> > running_dataset(dataset), residue(dataset);
    std::vector<std::vector<double> > data_grid(dataset);

    for(int i =0; i< data_grid.size();i++) data_grid[i].pop_back();
    for(int i= 0; i< n_est; i++){
        struct CART* new_tree = createCART(running_dataset,thresholdSamples,depth);
        res.push_back(new_tree);
        std::vector<double> y_hat = predict(*new_tree,data_grid);
        for(int j =0; j<data_grid.size();j++){
            residue[j][data_grid[0].size()] -= y_hat[j];
            running_dataset[j][data_grid[0].size()] = residue[j][data_grid[0].size()] * lr;
        }
        if(i%(std::max(1,n_est/10))==0)  std::cout<<i+1 << "  estimators trained" <<std::endl;
    }
    return res;
}


//Integrate the BDT
double BDT_evalutaor(std::vector<struct CART* > BDT,int dim){
    double res = 0;
    std::vector<std::vector<double> > initial_intervals;
    for(int i = 0; i < dim; i++){
        std::vector<double> cur_dim;
        cur_dim.push_back(0);
        cur_dim.push_back(1);
        initial_intervals.push_back(cur_dim);
    }
    for(int i = 0; i< BDT.size();i++){
        struct CART new_tree = * BDT[i];
        res += tree_evaluator(new_tree,initial_intervals);
    }
    return res;
}


//Final step, MC integration
void BDT_MC(std::vector<std::vector<double> > dataset, double ratio, int thresholdSamples = 4, int depth=2,int n_est = 10,double lr =0.1, int dim=5){
    int split = dataset.size() * lr;
    std::vector<std::vector<double> > train_dataset, test_dataset;
    for(int i = 0; i< split; i++) train_dataset.push_back(dataset[i]);
    for(int i = split; i < dataset.size(); i++) test_dataset.push_back(dataset[i]);
    std::vector<std::vector<double> > test_grid(test_dataset), test_val;
    for(int i = 0; i< test_grid.size();i++) test_grid[i].pop_back();

    std::vector<struct CART* > BDT = create_BDT(train_dataset,thresholdSamples,depth, n_est, lr);
    std::vector<double> y_hat,residue;
    double residue_sum = 0, residue_square_sum = 0;
    for(int i = 0;i < test_dataset.size();i++){
        double tmp_val = 0;
        for(int j = 0;j < n_est; j++) tmp_val += valuePredict(* (BDT[j]), test_grid[i]);
        y_hat.push_back(tmp_val);
        double residue_val_tmp = test_dataset[i][test_dataset[0].size()-1] - tmp_val;
        residue.push_back(residue_val_tmp);
        residue_sum += residue_val_tmp;
        residue_square_sum += residue_val_tmp*residue_val_tmp;
    }
    double residue_val = residue_sum / residue.size();
    double err = sqrt((residue_square_sum / residue.size() - residue_val*residue_val)/residue.size());
    double MC_value = BDT_evalutaor(BDT,dim);
    std::cout << "The integration value is " << MC_value << std::endl;
    std::cout << "The error is " << err <<std::endl;
    return;
}


//Print tree
void print_tree(struct CART tree_node){
    if(tree_node.is_leaf){
        std::cout <<"This is leave, the value is " << tree_node.value <<std::endl;
        return;
    }
    std::cout << "This is a split node, split index is " << tree_node.bestSplitFeature << " ,split value is " << tree_node.bestSplitValue <<std::endl;
    std::cout << "Left child is " << std::endl;
    print_tree(* (tree_node.left_tree));
    std::cout << "Right child is " << std::endl;
    print_tree( * (tree_node.right_tree));
    return;
}

/*
The following is testing part.
 */

double Integrand(std::vector<double> grid){
    double res;
    double x,y,z,w;
    x = grid[0];
    y = grid[1];
    z = grid[2];
    z = grid[3];
    w = grid[4];
    res = x+y+z-x*y+y*y+ z*y*z;
    return res;
}


int main(){
    srand(time(0));
    double val_sum = 0, val_square_sum = 0;
    //Creating testing grid
    std::vector<std::vector<double> > grid;
    for(int i = 0; i < 100000; i++){
        std::vector<double> tmp_point;
        for(int j = 0; j< 5; j++){
            double r =  ((double) rand() / (RAND_MAX));
            tmp_point.push_back(r);
        }
       double val = Integrand(tmp_point);
       val_sum += val;
       val_square_sum += val*val;
       tmp_point.push_back(val);
        grid.push_back(tmp_point);
    }
    double naive_MC = val_sum/grid.size();
    double naive_err = sqrt((val_square_sum/grid.size() - naive_MC* naive_MC)/grid.size());
    std::cout << "Naive result is " << naive_MC <<"  Naive error is " << naive_err <<std::endl;


    int dim = grid[0].size()-1;

    //Prepare initial integration domain
    std::vector<std::vector<double> > initial_intervals;
    for(int i = 0; i < dim; i++){
        std::vector<double> cur_dim;
        cur_dim.push_back(0);
        cur_dim.push_back(1);
        initial_intervals.push_back(cur_dim);
    }

    //Create CART
    int tree_depth = 5;
    int thresholdSamples = 0;

/*
    struct CART* regressTree = createCART(grid,thresholdSamples,tree_depth);
    double res1 = tree_evaluator(*regressTree, initial_intervals);
    
    std::vector<std::vector<double> > original_grid(grid);
    for(int i =0; i< grid.size();i++) original_grid[i].pop_back();
    std::vector<double> y_hat = predict(*regressTree,original_grid);
    
    print_tree(*regressTree);
*/
    //Create BDT
    int n_est = 100;
    double lr = 0.1;
//    std::vector<struct CART* > BDT = create_BDT(grid,thresholdSamples,tree_depth, n_est, lr);
//    double res2 = BDT_evalutaor(BDT,dim);

    //Print the result of the integrations
//    std::cout << "Integration of a single Decision tree is " << res1 << std::endl;

//    std::cout << "Integration of the BDT is " << res2 << std::endl;
   


    BDT_MC(grid,0.1,thresholdSamples,tree_depth,n_est,lr,dim); 
    return 0;


}
