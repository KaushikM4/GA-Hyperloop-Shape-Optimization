#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
using namespace std;
const int num_points = 80;  //Put even number
double A_throat = 1.0;
double gamma = 1.41;
double Mach_a = 0.4;
double pa = 100;    // in kPa

class Control{
    public:
        double x;
        double y;

};

double ex = (gamma + 1) / (2 * (gamma - 1));

double K = pow(0.5 * (gamma + 1), -ex);

double P_0a = pa * pow((1 + 0.5*(gamma  - 1)*pow(Mach_a, 2)), gamma/(gamma - 1));

class P{

    public:
        double x;
        double y;

        double A ;

        double ratio;

        double alpha;
        double M;
        double pressure;

};
int fact(int n){
        return tgamma(n+1);
}

double Bezier_x(Control *Pi, int n, double t){
    double P_x=0;
    for(int i=0;i<=n;i++){
        P_x+=(Pi[i].x)*(fact(n)*pow(1-t,n-i)*pow(t,i))/(fact(n-i)*fact(i));
    }

    return P_x;

}
double Bezier_y(Control *Pi, int n, double t){
    double P_y=0;
    for(int i=0;i<=n;i++){
        P_y+=(Pi[i].y)*(fact(n)*pow(1-t,n-i)*pow(t,i))/(fact(n-i)*fact(i));
    }

    return P_y;

}
int main(){
        int n=3;//degree of Bezier Curve
        Control Pi[n+1];

        Control& P_v=Pi[1]; //control point to be varied

        Pi[0].x = 0.0;
        Pi[0].y = 0.0;

        Pi[1].x=1.0;
        Pi[1].y=1.0;

        //Pi[1].x =1.5;
        //Pi[1].y =1.0803158617532618;

        Pi[2].x=1.0;
        Pi[2].y=2.0;

        Pi[2].x =1.45678645;
        Pi[2].y =1.819817499348621;

        Pi[3].x = 0.0;
        Pi[3].y = 3.0;


        string filename;
        filename="P2.txt";
        ifstream infile(filename);
        if (infile.is_open()) {
        infile >> P_v.x >> P_v.y;
        infile.close();
        //cout << "P1.x = " << P1.x << ", y = " << P1.y << endl;
        } 
        //else 
        //cout << "Unable to open file" << endl;

        P point[num_points];

        //Find Coordinates of all the necessary evaluation points
        for(int i = 0; i<num_points; i++){

            double t = (double)i/ (double)(num_points-1);
            point[i].x = Bezier_x(Pi,n, t);
            point[i].y = Bezier_y(Pi,n, t);

        }

        //Compute pressure at all these points

        for (int i = 0; i < num_points; i++)
        {            
            if(i < num_points/2){
                point[i].A = point[i].y + A_throat;
                point[i].ratio = point[i].A / A_throat;
            }

            else{
                point[i].A = A_throat + (Pi[3].y - point[i].y);
                point[i].ratio = point[i].A / A_throat;
            }

            point[i].alpha = pow(point[i].ratio / K, (1 / ex));
            point[i].M = (point[i].alpha - sqrt(point[i].alpha * point[i].alpha - 2 * (gamma - 1))) / (gamma - 1);
            point[i].pressure = P_0a / (pow((1 + 0.5 * (gamma - 1) * pow(point[i].M, 2)), gamma / (gamma - 1)));

        }

        //Compute Drag by summing pressure*Dy
        double drag = 0.0;

        for(int i = 0; i<(num_points); i++){

            drag += (point[i].pressure) * (point[i].y - point[i - 1].y);
        }

        //drag = 2*drag;
        cout <<drag;

}