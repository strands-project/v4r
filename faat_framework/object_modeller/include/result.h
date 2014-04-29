#pragma once

#include <vector>

namespace object_modeller
{

template<class T>
class ResultBase
{
public:
    virtual T getResult() = 0;

    operator T()
    {
        return getResult();
    }
};

template<class T>
class Result;

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

    TARGETTYPE getResult()
    {
        return boost::tuples::get<INDEX>(source->getResult());
    }
};

template<class T>
class Result : public ResultBase<T>
{
private:
    T result;
public:
    Result() {}
    Result(T result) {
        setResult(result);
    }

    void setResult(T result)
    {
        this->result = result;
    }
    T getResult()
    {
        return result;
    }

    template<int INDEX>
    ResultBase<typename boost::tuples::element<INDEX, T>::type> * get()
    {
        return new ResultWrapper<INDEX, typename boost::tuples::element<INDEX, T>::type, T>(this);
    }
};

template<class T, class U>
class ResultAggregator : public ResultBase<T>
{
private:
    U *input;
public:
    ResultAggregator(U *input)
    {
        this->input = input;
    }

    T getResult() // T is eg. tuple<string, int>
    {
        T result(*input);
        return result;
    }

};

}
