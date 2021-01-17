#include<string>
#include<iostream>
#include<sstream>

class Record {
public:
    std::string modelName1;
    int l1;
    int r1;
    int bs1;
    std::string modelName2;
    int l2;
    int r2;
    int bs2;
    float result1;
    float result2;
    float result;
    Record(std::string mn1_, int l1_, int r1_, int bs1_, std::string mn2_, int l2_, int r2_, int bs2_, float res1, float res2, float res) :
        modelName1(mn1_), l1(l1_), r1(r1_), bs1(bs1_),modelName2(mn2_), l2(l2_), r2(r2_), bs2(bs2_), result1(res1), result2(res2), result(res) {}
    
    std::string toString() {
        std::stringstream is;
        is<<modelName1<<","<<l1<<","<<r1<<","<<bs1<<","<<modelName2<<","<<l2<<","<<r2<<","<<bs2<<","<<result1<<","<<result2<<","<<result<<std::endl;
        std::string str;
        is>>str;
        return str;
    }
};