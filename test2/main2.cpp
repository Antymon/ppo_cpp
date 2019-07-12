//
// Created by szymon on 12/07/19.
//

/* C++ Program to illustrates the use of Constructors in multilevel inheritance  */

#include<iostream>
using namespace std;

class A
{
public:
    A(int x) {
        cout << "\nA: " << x ;
    }
};

class B : public virtual A
{
public:
    B(int y): A(y) {
        cout << "\nB: " << y ;
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

    return 0;
}