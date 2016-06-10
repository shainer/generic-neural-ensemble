#include <Utils.h>
#include <cstdlib>
#include <cmath>
#include <iostream>

#define PI 3.14

int randomInteger(int min, int max)
{
    if (max - min <= 0) {
        std::cerr << "Errore nella generazione casuale: " << max << " e " << min << std::endl;
        exit(0);
    }
    
    int r = min + (rand() % (max - min));
    return r;
}

double randomDouble(double min, double max)
{
    double f = (double)rand() / (double)RAND_MAX;
    return min + f * (max - min);
}

double max(double x1, double x2)
{
    if (x1 > x2) {
        return x1;
    }
    
    return x2;
}

double minimum(double a, double b)
{
    return (a < b) ? a : b;
}

double normalDistribution(double x, double sigma, double mu)
{
    double factor = 1 / (sigma * sqrt(2 * PI));
    double exponent = -0.5 * pow((x - mu) / sigma, 2);
    
    return factor * exp(exponent);
}

double gaussianMutation(double x, double a, double b)
{
    double sigma = b - a;
    double mu = (b - a) / 2;
    
    double distrib = normalDistribution(x, sigma, mu);
    
    return minimum(max(distrib, a), b);
}
