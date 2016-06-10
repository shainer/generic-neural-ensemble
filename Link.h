/*
 * This class defines a link between two neurons, called predecessor and successor
 */

#ifndef LINK_H
#define LINK_H

class Neuron;

class Link
{
public:
    explicit Link(double, Neuron *, Neuron *);
    virtual ~Link();
    
    /*
     * Returns the weight of this link.
     */
    double weight() const;
    
    /*
     * The output of a link is actually the output of the predecessor neuron in the last computation. Here
     * it can be easily accessed by the successor neuron.
     */
    double output() const;
    
    /*
     * Gradients are stored here, since they are related to the weights.
     */
    double gradient() const;
    double previousGradient() const;
    
    /*
     * This is the last "delta" applied to this link and computed by the RPROP algorithm: it will be needed in
     * the next application of rprop.
     */
    double delta() const;
    
    Neuron* successor() const;
    Neuron* predecessor() const;
    
    void setWeight(double);
    void setOutput(double);
    void setGradient(double);
    void setDelta(double);
    
private:
    int m_id;
    double m_weight;
    double m_output;
    double m_gradient, m_prevGradient, m_delta;
    
    Neuron* m_next;
    Neuron* m_prev;
};

#endif