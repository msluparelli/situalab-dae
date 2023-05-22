package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.sqrt;
import static java.lang.StrictMath.pow;



public class dae_thetas implements Serializable {

    public double[][] getThetasAE(int thetasrow, int thetascol){
        //deeplab_thetas
        int thetaslen = (thetasrow+1)*thetascol; //deeplab_thetas count

        //random Initialization
        Random rand = new Random();rand.setSeed(45632988);rand.longs(thetaslen);

        //deeplearning random distribution
        double factor = 1.0;double n = thetasrow;double stdev = sqrt(factor/n);
        double[] thetasDeepLearning = new double[thetaslen];
        for (int i = 0; i < thetaslen; i++) {
            double wfm = rand.nextGaussian()*stdev;
            thetasDeepLearning[i] = wfm;
        }

        //thetasList
        double[] thetasArray = Arrays.copyOfRange(thetasDeepLearning, 0, thetaslen); //deeplab_thetas
        int ij = 0;
        double[][] thetaslayer = new double[thetasrow+1][thetascol];
        for (int j=0; j<thetaslayer[0].length; j++){
            for (int i=0; i<thetaslayer.length; i++) {
                thetaslayer[i][j] = thetasArray[ij];
                if (i == 0) {
                    thetaslayer[0][j] = thetasArray[ij] * 0;
                } //bias init at zero (ReLU, 0.1)
                ij += 1;
            }
        }

        return thetaslayer;
    }

    //get thetas Transpose List
    public List<double[][]> getthetasTList(List<double[][]> thetasL){
        List<double[][]> thetasTL = new ArrayList<double[][]>();
        for (int layer=0; layer<thetasL.size(); layer++){
            double[][] thetasTlayer = getTransposeAE(thetasL.get(layer));
            thetasTL.add(thetasTlayer);
        }
        return thetasTL;
    }

    //transpose Matrix
    public double[][] getTransposeAE(double[][] initMatrix){
        int row = initMatrix.length;
        int col = initMatrix[0].length;
        double[][] outputMatrix = new double[col][row];
        for (int j=0; j<col; j++) {
            for (int i = 0; i<row; i++) {
                outputMatrix[j][i] = initMatrix[i][j];
            }
        }
        return outputMatrix;
    }

    //thetasStar, thetas with an additional bias unit for decoder
    public double[][] getthetasStar (double[][] thetas){

        double[][] thetasStar = new double[thetas.length][thetas[0].length+1];
        for (int row=0; row<thetasStar.length; row++){
            for (int col=0; col<thetasStar[0].length; col++){
                if (col==0) {
                    thetasStar[row][col] = 0;
                } else {
                    thetasStar[row][col] = thetas[row][col-1];
                }
            }
        }

        return thetasStar;
    }

    public double[][] getthetasS (double[][] thetas){

        int row = thetas[0].length+1;
        int col = thetas.length-1;
        double[][] thetasStar = new double[row][col];
        for (int j=1; j<col+1; j++) {
            for (int i=1; i<row; i++) {
                //System.out.println(thetas[j][i-1]);
                //System.out.println(i+" "+j+" "+thetas[j][i-1]);
                thetasStar[i][j-1] = thetas[j][i-1];
            }
        }
        return thetasStar;

    }


    //flatten Matrix
    public double[] getflatthetasAE(List<double[][]> thetasL){
        double[] thetasflat = new double[0];
        for (int layer=0; layer<thetasL.size(); layer++){
            double[] tflat = Arrays.stream(thetasL.get(layer)).flatMapToDouble(Arrays::stream).toArray();
            thetasflat = ArrayUtils.addAll(thetasflat, tflat);
        }
        return thetasflat;
    }

    //get square roots of weights
    public double getthetassqw(double[][] thetas){
        double sqw = 0;
        for(int i=1; i<thetas.length; i++){
            for (int j=0; j<thetas[0].length; j++){
                sqw += pow(thetas[i][j],2); //bias is always zero
            }
        }
        return sqw;
    }



    public void printMatrix(double[][] matrixPrint){

        System.out.print("\nprint matrix method row(i):"+matrixPrint.length+" col(j):"+matrixPrint[0].length+"\n");
        for (int row = 0; row < matrixPrint.length; row++) {
            for (int column = 0; column < matrixPrint[row].length; column++) {
                System.out.print(matrixPrint[row][column] + " ");
            }
            System.out.println();
        }

    }

    public void printMatrixdim(double[][] matrix){
        System.out.println("\nrow(i):"+matrix.length+" col(j):"+matrix[0].length);
    }




}
