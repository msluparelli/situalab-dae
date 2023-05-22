package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.DoubleAccumulator;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.log;

public class dae_feedback_regular implements Serializable {

    //method
    public void getAutoencoders(LabeledPoint xvalues,
                                List<double[][]> thetasL,
                                List<double[][]> thetasTL,
                                double[][] thetasS,
                                double[][] thetasST,
                                dae_AccumM GthetaAEAcc,
                                dae_AccumM GthetaSAEAcc,
                                DoubleAccumulator JthetasCost,
                                DoubleAccumulator mAE,
                                String AEmethod) throws IOException {

        dae_thetas AEthetas = new dae_thetas();

        //feature init
        double[] biasUnit = {1}; double[] zi = new double[1]; double[] zout; double[] xh;

        //x values
        double[] xvalue = xvalues.features().toArray(); //x values
        double[] xinput = new double[xvalue.length]; //init xinput, used to get gradient (xhat-xinput)


        //feedforward, hiddenLayer
        double[] ai = ArrayUtils.addAll(biasUnit, xvalue); //xinput
        double[] aitilde = ai; //used for denoising/constrain AE
        for (int hiddenLayer=0; hiddenLayer<thetasL.size(); hiddenLayer++){
            xinput = ai; //fix input value to get xhat-xinput of the last hiddenlayer
            aitilde = ai; //fix aitilde
            aitilde = getvaluestilde(ai, aitilde, AEmethod); //regular/denoising/constrain AE
            zi = multiMV(thetasTL.get(hiddenLayer), aitilde); //zetas from regular/denoising/constrain AE
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

        getBackPropagationAE(ejxh, ai, zi, thetasS, xinput, GthetaAEAcc, GthetaSAEAcc);

        mAE.add(1.0);

    }


    //backpropagation algorithm
    public void getBackPropagationAE(double[] ej,
                                     double[] ai,
                                     double[] zi,
                                     double[][] thetasS,
                                     double[] xi,
                                     dae_AccumM GthetaAEAcc,
                                     dae_AccumM GthetaSAEAcc){


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
        getEjAE(ejencoder, thetasS, ej, dziBPAE); //get ejl{thetasS, ejdecoder, z(ai)}

        //get gradient thetas
        ej = Arrays.copyOfRange(ejencoder,1, ejencoder.length); // ej(l), remove bias unit
        double[][] gthetahiddenlayer = new double[xi.length][ej.length]; //init gradient matrix
        getgradientthetasAE(gthetahiddenlayer, xi, ej); //get gradient
        GthetaAEAcc.add(gthetahiddenlayer); //gradient accumulator works fine

    }

    public double[] getvaluestilde (double[] xvalue,
                                    double[] xtilde,
                                    String AEmethod){

        if (AEmethod.equals("denoising")) {

            Random rand = new Random();
            rand.setSeed(76835279);
            rand.longs(xvalue.length);
            for (int r=1; r<xvalue.length; r++){
                double rindx = 0;
                double rvalue = rand.nextDouble();
                if (rvalue>=0.5) {rindx = 1;}
                xtilde[r] = xtilde[r]*rindx; //denoising autoencoder
            }
        }

        return xtilde;
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
    public void getEjAE(double[] ejout, double[][] Wij, double[] ej, double[] dzi) {
        for (int row = 0; row < Wij.length; row++) {
            double rowSum = 0;
            for (int col = 0; col < Wij[row].length; col++) {
                rowSum += (Wij[row][col]*ej[col]);
            }
            ejout[row] = rowSum*dzi[row];
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
