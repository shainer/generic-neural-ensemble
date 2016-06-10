/*
 * Utility functions for all the program
 */

#ifndef UTILS_H
#define UTILS_H

#include <QString>

/*
 * Defines the possible mutation we apply to the networks
 */
enum MutationOperator
{
    RemoveLink = 1,
    AddLink,
    RemoveNeuron,
    AddNeuron,
    WeightMutation
};

/*
 * Random number generation
 */
int randomInteger(int, int);
double randomDouble(double, double);

/*
 * Returns the new weight on which we applied a Gaussian mutation.
 */
double gaussianMutation(double, double, double);
double minimum(double, double);

#endif