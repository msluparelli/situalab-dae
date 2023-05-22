package com.situalab.dlab;

import org.apache.spark.util.AccumulatorV2;

import java.io.Serializable;


public class dae_AccumV extends AccumulatorV2<double[], double[]> implements Serializable {

    //attributes
    private double[] aiA;
    private int iA;

    //constructor
    public dae_AccumV(int i){
        double[] aiA_ = new double[i];
        this.aiA = aiA_;
        this.iA = i; //para utilizar en copy y reset
    }

    @Override
    public boolean isZero() {
        return true;
    }

    @Override
    public AccumulatorV2<double[], double[]> copy() {
        return (new dae_AccumV(iA));
    }

    @Override
    public void reset() {
        aiA = new double[iA];

    }

    @Override
    public void add(double[] dbias) {
        for (int row=0; row<dbias.length; row++) {
            aiA[row] += dbias[row];
        }

    }

    @Override
    public void merge(AccumulatorV2<double[], double[]> other) {
        add(other.value());

    }

    @Override
    public double[] value() {
        return aiA;
    }

}
