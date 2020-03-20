
#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2.h"
#include "TTree.h"
#include <iostream>
#include "TEfficiency.h"
#include "TStyle.h"
#include "../interface/helpers.h"

#include "../interface/globals.h"


int main(){

    gStyle->SetOptStat(0);
    global::setTrees();

    compareJetMass jm_1_5  (41,0.6,1.1,"m_{j}(r)/m_{j}(t)","# events",1,6);
    compareJetMass jm_5_10 (41,0.6,1.1,"m_{j}(r)/m_{j}(t)","# events",6,11);
    compareJetMass jm_10_15(41,0.6,1.1,"m_{j}(r)/m_{j}(t)","# events",11,16);



    TCanvas *cv=createCanvas();
    legends::buildLegends();

    jm_1_5.setClassicLineColourAndStyle(-1,3);
    jm_5_10.setClassicLineColourAndStyle(-1,2);
    jm_1_5.setOCLineColourAndStyle(-1,3);
    jm_5_10.setOCLineColourAndStyle(-1,2);
   // jm_10_15.setClassicLineColourAndStyle(-1,1);


    jm_1_5.DrawAxes();
    jm_1_5.AxisHisto()->GetYaxis()->SetRangeUser(10, 20000);
    jm_1_5.Draw("same");
    jm_5_10.Draw("same");
    jm_10_15.Draw("same");


    placeLegend(legends::legend_full, 0.2, 0.5)->Draw("same");

    cv->SetLogy();

    cv->Print("jetmass.pdf");

}
