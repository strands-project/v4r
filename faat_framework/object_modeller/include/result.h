#pragma once

#include <vector>
#include <map>

#include <boost/tuple/tuple.hpp>

namespace object_modeller
{

template<class T>
class ResultBase
{
protected:
    ResultBase<T> *fallback;
public:
    ResultBase()
    {
        fallback = NULL;
    }

    void setFallback(ResultBase<T> *fallback)
    {
        if (this->fallback == NULL)
        {
            this->fallback = fallback;
        }
        else
        {
            this->fallback->setFallback(fallback);
        }
    }

    virtual T getResult(int sequenceId) = 0;
    virtual int getNrSequences() = 0;
};

template<class T>
class Result;

/*
template<int INDEX, class TARGETTYPE, class SOURCETYPE>
class ResultWrapper : public ResultBase<TARGETTYPE>
{
private:
    Result<SOURCETYPE> *source;
public:
    ResultWrapper(Result<SOURCETYPE> *source)
    {
        this->source = source;
    }

    TARGETTYPE getResult(int sequenceId)
    {
        return boost::tuples::get<INDEX>(source->getResult(sequenceId));
    }

    virtual int getNrSequences()
    {
        return source->getNrSequences();
    }
};
*/

struct ResultException : public std::exception
{
   std::string s;
   ResultException(std::string ss) : s(ss) {}
   ~ResultException() throw () {} // Updated
   const char* what() const throw() { return s.c_str(); }
};

template<class T>
class Result : public ResultBase<T>
{
private:
    std::map<int, T> result;
public:
    Result() {}
    /*
    Result(T result) {
        //setResult(result);
    }
    */

    void setResult(int sequenceId, T result)
    {
        this->result[sequenceId] = result;
    }
    T getResult(int sequenceId)
    {
        if (result.count(sequenceId) != 0)
        {
            return result[sequenceId];
        }

        if (this->fallback != NULL)
        {
            std::cout << "no result found, using fallback" << std::endl;
            return this->fallback->getResult(sequenceId);
        }

        throw ResultException("WARNING: accessing nonexistent element");
    }

    virtual int getNrSequences()
    {
        if (result.size() == 0 && this->fallback != NULL)
        {
            return this->fallback->getNrSequences();
        }

        return result.size();
    }

    /*
    template<int INDEX>
    ResultBase<typename boost::tuples::element<INDEX, T>::type> * get()
    {
        return new ResultWrapper<INDEX, typename boost::tuples::element<INDEX, T>::type, T>(this);
    }
    */
};


template<class T>
class CustomResult : public ResultBase<T>
{
private:
    T result;
public:
    CustomResult(T result) {
        this->result = result;
    }
    T getResult(int sequenceId)
    {
        return result;
    }

    virtual int getNrSequences()
    {
        return 1;
    }
};



}
