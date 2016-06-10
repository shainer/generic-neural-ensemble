#include <ProblemInfo.h>
#include <Utils.h>
#include <QDir>

#include <iostream>

using namespace std;

class ProblemInfoHelper
{
public:
    ProblemInfoHelper()
        : q(0)
    {}
    
    virtual ~ProblemInfoHelper()
    {
        delete q;
    }
    
    ProblemInfo* q;
};

Q_GLOBAL_STATIC(ProblemInfoHelper, s_probleminfo);

ProblemInfo::ProblemInfo()
{
    Q_ASSERT (!s_probleminfo()->q);
    
    readSamples("../tictactoe");
    
    s_probleminfo()->q = this;
}

ProblemInfo::~ProblemInfo()
{}

ProblemInfo* ProblemInfo::instance()
{
    if (!s_probleminfo()->q) {
        new ProblemInfo;
    }
    
    return s_probleminfo()->q;
}

void ProblemInfo::destroy()
{
    qDeleteAll(m_training);
    qDeleteAll(m_test);
    
    m_training.clear();
    m_test.clear();
}

QList< InputSample* > ProblemInfo::trainingSamples() const
{
    return m_training;
}

QList< InputSample* > ProblemInfo::testSamples() const
{
    return m_test;
}

void ProblemInfo::readSamples(const QString& dir)
{
    QDir sampleDir(dir);
    QList< InputSample* > samples;
    
    if (!sampleDir.exists()) {
        std::cerr << "Sample directory " << dir.toStdString() << " doesn't exist." << std::endl;
        exit(-1);
    }
    
    QFile sampleFile( sampleDir.absoluteFilePath("tic-tac-toe.data") );
    sampleFile.open(QFile::ReadOnly);
    
    while (true) {
        QString line( sampleFile.readLine() );
        
        if (line.isEmpty()) {
            break;
        }
        
        InputSample* sample = new InputSample;
        int index = 0;
        
        Q_FOREACH (const QString& piece, line.split(',')) {            
            if (piece == "x") {
                sample->attributes[index] = -1.0;
            } else if (piece == "b") {
                sample->attributes[index] = 0.0;
            } else if (piece == "o") {
                sample->attributes[index] = 1.0;
            } else if (piece == "positive\n") {
                sample->n_class = 1;
            } else if (piece == "negative\n") {
                sample->n_class = 0;
            } else {
                cerr << "Error while reading: unrecognized data \"" << piece.toStdString() << "\"" << endl;
                qDeleteAll(samples);
                sampleFile.close();
                exit(-1);
            }
            
            index++;
        }
        
        samples.prepend(sample);
    }
    
    sampleFile.close();
    qSort(samples.begin(), samples.end(), ProblemInfo::randomOrder);
    
    m_training = samples.mid(0, 600);
    m_test = samples.mid(601);
}

bool ProblemInfo::randomOrder(const InputSample* s1, const InputSample* s2)
{
    Q_UNUSED(s1)
    Q_UNUSED(s2)
    
    double choice = randomDouble(0.0, 1.0);
    return (choice <= 0.5);
}