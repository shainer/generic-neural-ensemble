/*
 * This header contains macro and classes that hold general information about the problem being solved.
 */

#ifndef PROBLEMINFO_H
#define PROBLEMINFO_H

#include <QString>
#include <QHash>

/* Size of the input feature vector (and number of neurons in the input layer) */
#define INPUT_SIZE  9

/* Size of the output (and number of neurons in the output layer) */
#define OUTPUT_SIZE 1 /* being a 2-class classification problem, I only need one */

/* Number of different classes */
#define NUM_CLASSES 2

/* Number of neurons on the hidden layer */
#define HIDDEN_SIZE     10
#define HIDDEN_SIZE_MAX 10
#define HIDDEN_SIZE_MIN 4

#define LINK_SIZE_MIN   15

/* RPROP parameters */
#define POSITIVE_ETA 10.2
#define NEGATIVE_ETA 0.001
#define MAX_STEP     50.0
#define MIN_STEP     0.0
#define INITIAL_STEP 0.0125

/*
 * Holds a sample for this problem. Each sample is composed by an attribute vector and
 * the class it belongs to.
 */
class InputSample
{
public:
    InputSample()
        : n_class(0)
    {
        for (int attr = 0; attr < INPUT_SIZE; attr++) {
            attributes[attr] = 0.0;
        }
    }
    
    double attributes[INPUT_SIZE];
    unsigned int n_class;
};

/*
 * Manages the samples for the problem
 */
class ProblemInfo
{
    /*
     * Disables the default copy constructor (only one instance can be present)
     */
    Q_DISABLE_COPY(ProblemInfo)
    
public:
    virtual ~ProblemInfo();
    
    /*
     * Returns an unique instance of this class.
     * First invocation has to read the samples from the hard-disk, so it may be slow.
     */
    static ProblemInfo* instance();
    
    /*
     * Destroys the instance.
     */
    void destroy();
    
    /*
     * Returns all the training samples.
     */
    QList< InputSample* > trainingSamples() const;
    
    /*
     * Returns all the test samples.
     */
    QList< InputSample* > testSamples() const;
    
private:
    ProblemInfo();
    
    void readSamples(const QString &);
    
    /*
     * Utility function used to randomly permute the samples in our lists. It must be static
     * because it's used as a sort function to qSort.
     */
    static bool randomOrder(const InputSample *, const InputSample *);
    
    QList< InputSample* > m_training;
    QList< InputSample* > m_test;
};

#endif