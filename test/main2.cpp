//
// Created by szymon on 12/07/19.
//

/* C++ Program to illustrates the use of Constructors in multilevel inheritance  */

#include<iostream>
#include <vector>
#include <algorithm>


using namespace std;

class A
{
public:
    int x;

    A(int x): x{x} {
        cout << "\nA: " << x ;
    }
};

class B : public virtual A
{
public:
    B(int y): A(y) {
        cout << "\nB: " << y ;
        cout << "\nBx: " << x ;
    }
};

class C : public virtual B
{
public:
    C(int x, int y) : A(x), B(y){}
};

int main()
{
    // B's initialization of A will get ignored!!!
    C c(5, 7) ;

    int n_batch = 5;
    int noptepochs = 20;

    std::vector<int> inds(n_batch);
    for (int i=0; i<n_batch; ++i) {
        inds[i]=i;
    }
    for (int epoch_num = 0; epoch_num<noptepochs; epoch_num++) {
        std::random_shuffle(inds.begin(), inds.end());
        std::cout << "\nv: ";
        for (auto iv: inds) {
            std::cout << iv << " ";
        }
        std::cout << "\n";
    }

    return 0;
}