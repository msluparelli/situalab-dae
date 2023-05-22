package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.DoubleAccumulator;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.log;


public class dae_feedback_sparse implements Serializable {

    //method
    public void getAverageActivation(LabeledPoint xvalues,
                                     List<double[][]> thetasTL,
                                     dae_AccumV pjAESparseAcc,
                                     DoubleAccumulator mAE){

        dae_thetas AEthetas = new dae_thetas();

        //feature init
        double[] biasUnit = {1}; double[] zi = new double[1]; double[] axhat; double[] xhat;
        double[] xinput = new double[1]; //init xinput, used to get gradient (xhat-xinput)
        int lastthetas = thetasTL.size()-1; //last thetas to backpropagate
        List<double[]> A = new ArrayList<double[]>();


        //x values
        double[] xvalue = xvalues.features().toArray(); //x values


        //feedforward, hiddenLayer
        double[] ai = ArrayUtils.addAll(biasUnit, xvalue); //xinput
        for (int hiddenLayer=0; hiddenLayer<thetasTL.size(); hiddenLayer++){
            xinput = ai; //fix input value to get xhat-xinput of the last hiddenlayer
            zi = multiMV(thetasTL.get(hiddenLayer), xinput); //zetas from regular/denoising/constrain AE
            ai = Arrays.stream(zi).map(xoz -> dae_activation.sigmoid.apply(xoz)).toArray(); //encoder, sigmoid function
            ai = ArrayUtils.addAll(biasUnit, ai); //add bias for next hiddenLayer
        }

        //accumulate activations per hidden layer // avoid biasUnit term
        double[] pj = Arrays.copyOfRange(ai, 1, ai.length); //include bias from:0, otherwise from:1
        pjAESparseAcc.add(pj);
        mAE.add(1.0);

    }




    public void getAutoencodersSparse(LabeledPoint xvalues,
                                      List<double[][]> thetasL,
                                      List<double[][]> thetasTL,
                                      double[][] thetasS,
                                      double[][] thetasST,
                                      dae_AccumM GthetaAEAcc,
                                      dae_AccumM GthetaSAEAcc,
                                      DoubleAccumulator JthetasCost,
                                      double[] KLg, double beta) throws IOException {

        //ae_thetas AEthetas = new ae_thetas();

        //feature init
        double[] biasUnit = {1}; double[] zi = new double[1]; double[] zout; double[] xh;


        //x values
        double[] xvalue = xvalues.features().toArray(); //x values
        double[] xinput = new double[xvalue.length]; //init xinput, used to get gradient (xhat-xinput)


        //feedforward, hiddenLayer
        double[] ai = ArrayUtils.addAll(biasUnit, xvalue); //xinput
        for (int hiddenLayer=0; hiddenLayer<thetasL.size(); hiddenLayer++){
            xinput = ai; //fix input value to get xhat-xinput of the last hiddenlayer
            zi = multiMV(thetasTL.get(hiddenLayer), xinput); //zetas from regular/denoising/constrain AE
            ai = Arrays.stream(zi).map(xoz -> dae_activation.sigmoid.apply(xoz)).toArray(); //encoder, sigmoid function
            ai = ArrayUtils.addAll(biasUnit, ai); //add bias for next hiddenLayer
        }


        //hidden layer decoder, from now on only for the last thetas
        zout = multiMV(thetasST, ai); //last ai from last hiddenLayer trained
        xh = Arrays.stream(zout).map(xoz -> dae_activation.sigmoid.apply(xoz)).toArray(); //decoder, sigmoid function


        //get gradient & cost function
        double[] ejxh = new double[xh.length]; double loglCost = 0;
        for (int g=0; g<xh.length; g++){
            ejxh[g] = xh[g]-xinput[g]; //gradient bias no incluido
            loglCost += -(xinput[g]*log(xh[g]) + (1-xinput[g])*log(1-xh[g])); //bias no inlcuido
        }
        JthetasCost.add(loglCost);

        getBackPropagationAESparse(ejxh, ai, zi, thetasS, xinput, GthetaAEAcc, GthetaSAEAcc, KLg, beta);

    }


    //backpropagation algorithm
    public void getBackPropagationAESparse(double[] ej,
                                           double[] ai,
                                           double[] zi,
                                           double[][] thetasS,
                                           double[] xi,
                                           dae_AccumM GthetaAEAcc,
                                           dae_AccumM GthetaSAEAcc,
                                           double[] KLg,
                                           double beta){


        //get gradient thetasS
        double[][] gthetaoutputlayer = new double[ai.length][ej.length]; //init gradient thetaS matrix
        getgradientthetasAE(gthetaoutputlayer, ai, ej); //get gradient
        GthetaSAEAcc.add(gthetaoutputlayer); //gradient accumulator works fine

        //dzi
        double[] biasUnit = {1}; //bias unit
        double[] dziBPAE = Arrays.stream(zi).map(xoz -> dae_activation.dsigmoid.apply(xoz)).toArray(); //get dactivation z
        dziBPAE = ArrayUtils.addAll(biasUnit, dziBPAE); //deactivated ax, incluye bias unit

        //thetasStartT & ej hidden layer init
        double[] ejencoder = new double[(thetasS.length)]; //ej hidden layer

        //get error hidden layer
        getEjAESparse(ejencoder, thetasS, ej, dziBPAE, KLg, beta); //get ejl{thetasS, ejdecoder, KLgi,  z(ai)}


        //get gradient thetas
        ej = Arrays.copyOfRange(ejencoder,1, ejencoder.length); // ej(l), remove bias unit
        double[][] gthetahiddenlayer = new double[xi.length][ej.length]; //init gradient matrix
        getgradientthetasAE(gthetahiddenlayer, xi, ej); //get gradient
        GthetaAEAcc.add(gthetahiddenlayer); //gradient accumulator works fine


    }

    //get zi values (matrix, vector multplication)
    public double[] multiMV(double[][] M, double[] v){
        double[] Mv = new double[M.length];
        for (int row=0; row<M.length; row++){
            double rowSum = 0;
            for (int col=0; col<M[row].length; col++){
                rowSum += (M[row][col]*v[col]);
            }
            Mv[row] = rowSum;
        }
        return Mv;
    }




    //get ej
    public void getEjAESparse(double[] ejout, double[][] Wij, double[] ej, double[] dzi, double[] KLg, double beta) {
        double[] biaszero = {0};
        double[] KLg_biaszero = ArrayUtils.addAll(biaszero, KLg); //xinput
        for (int row = 0; row < Wij.length; row++) {
            double rowSum = 0;
            for (int col = 0; col < Wij[row].length; col++) {
                rowSum += (Wij[row][col]*ej[col]);
            }
            ejout[row] = (rowSum+(beta*KLg_biaszero[row]))*dzi[row];
        }
    }

    //get dthetaslayer
    public double[][] getgradientthetasAE(double[][] derivativethetasLAE,
                                          double[] aj,
                                          double[] ej){
        for (int row = 0; row < aj.length; row++) {
            for (int col = 0; col < ej.length; col++) {
                derivativethetasLAE[row][col] = aj[row]*ej[col];
            }
        }
        return derivativethetasLAE;
    }


}
