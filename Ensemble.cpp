/*
 * This defines and trains an ensemble of neural networks.
 */

#include <Ensemble.h>
#include <QDebug>

#include <iostream>

using namespace std;

/*
 * Creates the required amount of networks
 */
NetworkEnsemble::NetworkEnsemble(int numNetworks)
{
    int i = 1;
    
    for (; i <= numNetworks; i++) {
        Network* network = new Network(i);
        m_networks.append(network);
    }
    
    m_nextId = i + 1;
}

/*
 * Deallocates the network objects
 */
NetworkEnsemble::~NetworkEnsemble()
{
    qDeleteAll(m_networks);
}

void NetworkEnsemble::training(QList< InputSample* >& trainingSamples, QList< InputSample* >& generationTest)
{
    QList< Network* > archive;
    QList< Network* > population( m_networks );
    
    int desiredPopulationSize = population.size();
    int desiredArchiveSize = desiredPopulationSize / 2;
    
    QList< InputSample* > generationTraining = trainingSamples.mid(0, 100);

    for (int epoch = 1; epoch <= 100; epoch++) {
        int iteration = 1;
        
        cout << ":: Epoch " << epoch << " running." << endl;
        
        /*
         * Life-long training using rprop
         */
        Q_FOREACH (InputSample* sample, generationTraining) {
            
            Q_FOREACH (Network* net, population) {
                net->applyInput(sample->attributes, sample->n_class);
                
                /*
                 * At the first iteration the "previous gradient" isn't defined so we skip rprop in that case
                 */
                if (iteration > 1) {
                    net->updateByRProp();
                }
            }
            iteration++;
        }
        
        /*
         * Compute the new average errors, to be used as an objective function to minimize in the genetic algorithm.
         */
        Q_FOREACH (Network* net, population) {
            net->setAverageError( computeAverageError(net, generationTest) );
        }

        QMap< int, QList< Network* > > ranks = computeParetoFrontRank(population);
        QList< Network* > rest;
        archive.clear();
        
        /*
         * Everything that doesn't go into the archive is added to the "rest" list, to be deallocated later.
         */
        for (QMap< int, QList< Network* > >::iterator it = ranks.begin(); it != ranks.end(); it++) {
            QList< Network* > currentFront = it.value();
        
            if (archive.size() == desiredArchiveSize) {
                rest += currentFront;
                continue;
            }
            
            if (archive.size() + currentFront.size() > desiredArchiveSize) {
                int remainingSpace = archive.size() + currentFront.size() - desiredArchiveSize;
                
                QList< Network* > sorted = sortBySparsity(currentFront);
                archive += sorted.mid(remainingSpace + 1); /* takes the sparsest ones */
                rest = sorted.mid(0, remainingSpace);
            } else {
                archive += currentFront;
            }
        }
        
        qDeleteAll(rest);
        rest.clear();
        
        population = breed(archive);
        population += archive;
    }

    m_networks = paretoFront(population);
}

double NetworkEnsemble::computeAverageError(Network* net, const QList< InputSample* >& set)
{
    double percentageError = 0.0;
    int wrong = 0;
    
    Q_FOREACH (InputSample* sample, set) {
        net->applyInput(sample->attributes, sample->n_class);
        
        if (net->output() != sample->n_class) {
            wrong++;
        }
    }
    
    percentageError = (double)wrong / (double)set.size();
    return percentageError;
}

double NetworkEnsemble::test(QList< InputSample* >& testSamples)
{    
    int right = 0;
    int wrong = 0;
    /*
     * The total answer is the answer given by the maximum number of networks in the Pareto front.
     */
    for (QList< InputSample* >::iterator sample = testSamples.begin(); sample != testSamples.end(); sample++) {        
        int answers[NUM_CLASSES];
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            answers[i] = 0;
        }

        for (QList< Network* >::iterator it = m_networks.begin(); it != m_networks.end(); it++) {
            (*it)->applyInput((*sample)->attributes, (*sample)->n_class);
            int output = (*it)->output();
                        
            answers[output]++;
        }
        
        int max = 0;
        unsigned int maxClass = 0;
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (answers[i] > max) {
                max = answers[i];
                maxClass = i;
            }
        }
        
        if (maxClass == (*sample)->n_class) {
            right++;
        } else {
            wrong++;
        }
    }
    
    cout << ":: Test results: " << right << " right answers and " << wrong << " wrong ones " << endl;
    
    double rightPercentage = (double)right / (double)testSamples.size();
    return rightPercentage;
}

/*
 * The new population is generated by the previous one
 */
QList< Network* > NetworkEnsemble::breed(QList< Network* > parents)
{
    QList< Network* > children;
    
    Q_FOREACH (Network* parent, parents) {
        Network* child = new Network(parent, m_nextId++);
        
        for (int i = 1; i <= 10; i++) {
            MutationOperator mutation = (MutationOperator)(randomInteger(1, 5));
            child->mutate(mutation);
        }
        
        children.append(child);
    }
    
    return children;
}

QMap< int, QList< Network*> > NetworkEnsemble::computeParetoFrontRank(QList< Network* > population)
{
    QList< Network* > currentPopulation( population );
    QMap< int, QList< Network* > > rankList;
    
    int i = 1;
    while ( !currentPopulation.isEmpty() ) {
        rankList.insert(i, paretoFront(currentPopulation));

        Q_FOREACH (Network* net, rankList[i]) {
            for (QList< Network* >::iterator it = currentPopulation.begin(); it < currentPopulation.end(); it++) {
                
                if ((*it)->id() == net->id()) {
                    currentPopulation.erase(it);
                }
            }
        }

        i++;
    }
    
    return rankList;
}

QList< Network* > NetworkEnsemble::paretoFront(QList< Network* >& population)
{
    QList< Network* > front;
    
    Q_FOREACH (Network* net, population) {
        bool paretoDominator = true;
        
        for (QList< Network* >::iterator it = front.begin(); it < front.end(); it++) {
            if (paretoDominates(net, (*it))) {
                front.erase(it);
            } else if (paretoDominates((*it), net)) {
                paretoDominator = false;
                break;
            }
        }
        
        if (paretoDominator) {
            front.append(net);
        }
    }
    
    return front;
}

bool NetworkEnsemble::paretoDominates(Network* n1, Network* n2)
{
    if (n1->averageError() < n2->averageError() && n1->complexity() <= n2->complexity()) {
        return true;
    }
    
    if (n1->complexity() < n2->complexity() && n1->averageError() <= n2->averageError()) {
        return true;
    }
    
    return false;
}

QList< Network* > NetworkEnsemble::sortBySparsity(QList< Network* > front)
{
    Q_FOREACH (Network* net, front) {
        net->setSparsity(0);
    }
    
    qSort(front.begin(), front.end(), NetworkEnsemble::lessThanError);
    
    front.first()->setSparsity(INT_MAX);
    front.last()->setSparsity(INT_MAX);
    
    for (QList< Network* >::iterator it = front.begin() + 1; front.size() > 1 && it != front.end() - 1; it++) {        
        double prevObj = (*(it - 1))->averageError();
        double nextObj = (*(it + 1))->averageError();
        
        if ((*it)->sparsity() < INT_MAX) {
            (*it)->addSparsity( nextObj - prevObj );
        }
    }
    
    qSort(front.begin(), front.end(), NetworkEnsemble::lessThanComplexity);
    front.first()->setSparsity(INT_MAX);
    front.last()->setSparsity(INT_MAX);
    
    for (QList< Network* >::iterator it = front.begin() + 1; front.size() > 1 && it != front.end() - 1; it++) {
        int prevObj = (*(it - 1))->complexity();
        int nextObj = (*(it + 1))->complexity();
        
        if ((*it)->sparsity() < INT_MAX) {
            (*it)->addSparsity( nextObj - prevObj );
        }
    }
    
    qSort(front.begin(), front.end(), NetworkEnsemble::lessThanSparsity);
    return front;
}

bool NetworkEnsemble::lessThanError(const Network* n1, const Network* n2)
{
    return (n1->averageError() < n2->averageError());
}

bool NetworkEnsemble::lessThanComplexity(const Network* n1, const Network* n2)
{
    return (n1->complexity() < n2->complexity());
}

bool NetworkEnsemble::lessThanSparsity(const Network* n1, const Network* n2)
{
    return (n1->sparsity() < n2->sparsity());
}
