
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


TTree * global::classic_tree=0;
TTree * global::oc_tree=0;


int global::defaultOCColour = kOrange+2;
int global::defaultClassicColour = kAzure-5;


 TString global::defaultOCLabel="Condensation";
 TString global::defaultClassicLabel="Baseline PF";

TCanvas * createCanvas(){
    TCanvas * cv = new TCanvas();
    cv->SetBottomMargin(0.15);
    cv->SetLeftMargin(0.15);
    return cv;
}


int plotscript(int argc, char* argv[]){
    if(argc<3) return -1;

    TString classic_files = argv[1];
    TString oc_files = argv[2];

    TFile classic_f(classic_files,"READ");
    TFile oc_f(oc_files,"READ");

    global::classic_tree = (TTree*)classic_f.Get("tree");
    global::oc_tree = (TTree*)oc_f.Get("tree");

    if(!global::classic_tree || !global::oc_tree){
        std::cerr << "tree could not be read from either file: " << classic_files << " or " << oc_files << std::endl;
        return -1;
    }
    TCanvas *cv=createCanvas();

    /////////standard legends


    TLegend * leg_a = new TLegend(0.6,0.25, 0.85, 0.5);
        leg_a -> SetLineWidth(1);
        makeLegEntry(leg_a,global::defaultOCLabel,"l",global::defaultOCColour);
        makeLegEntry(leg_a,global::defaultClassicLabel,"l",global::defaultClassicColour);
        leg_a->AddEntry("","","");
        makeLegEntry(leg_a,"1-5 particles","l",kBlack,2);
        makeLegEntry(leg_a,"6-10 particles","l",kBlack,1);
        leg_a->Draw();


        TLegend * leg_a2 = new TLegend(0.6,0.25, 0.85, 0.55);
        leg_a2 -> SetLineWidth(1);
        makeLegEntry(leg_a2,global::defaultOCLabel,"l",global::defaultOCColour);
        makeLegEntry(leg_a2,global::defaultClassicLabel,"l",global::defaultClassicColour);
        leg_a2->AddEntry("","","");
        makeLegEntry(leg_a2,"1-5 particles","l",kBlack,3);
        makeLegEntry(leg_a2,"6-10 particles","l",kBlack,2);
        makeLegEntry(leg_a2,"10-15 particles","l",kBlack,1);
        leg_a2->Draw();

////

    /////

    gStyle->SetOptStat(0);

    compareEfficiency eff_energy1_5("true_e", "is_true && n_true <= 5", "(is_true && is_reco && n_true <= 5)",5,0,200,"Momentum [GeV]","Efficiency");
    compareEfficiency eff_energy5_10("true_e", "is_true && n_true <= 10 && n_true>5", "(is_true && is_reco && n_true <= 10 && n_true>5)",5,0,200,"Momentum [GeV]","Efficiency");
    std::vector<compareEfficiency*> eff_mom = {&eff_energy1_5, &eff_energy5_10};



    eff_energy1_5.DrawAxes();
    eff_energy1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.75,1.01);

    for(int i=0;i<eff_mom.size();i++){
        eff_mom.at(i)->setOCLineColourAndStyle(-1,2-i);
        eff_mom.at(i)->setClassicLineColourAndStyle(-1,2-i);
        eff_mom.at(i)->Draw("same,P");
    }


    leg_a->Draw("same");
    cv->Print("mom_efficiency.pdf");



    ///////////////////////////////////////////

    compareEfficiency eff_n_true("n_true", "is_true", "(is_true && is_reco)",15,0.5,15.5,"Particles per event","Efficiency");
    eff_n_true.DrawAxes();
    eff_n_true.AxisHisto()->GetYaxis()->SetRangeUser(0.75,1.01);
    eff_n_true.Draw("same,P");


    TLegend * leg_b = new TLegend(0.2,0.27, 0.5, 0.5);
    leg_b -> SetLineWidth(1);
    makeLegEntry(leg_b,global::defaultOCLabel,"l",global::defaultOCColour);
    makeLegEntry(leg_b,global::defaultClassicLabel,"l",global::defaultClassicColour);
    leg_b->Draw("same");

    cv->Print("N_efficiency.pdf");

    ///////////////////////////////////////////


    compareTH1D resolution1_5("reco_e/true_e","is_true && is_reco && n_true <= 5",51,0.8,1.2,"p(r)/p(t)","# particles");
    compareTH1D resolution5_10("reco_e/true_e","is_true && is_reco && n_true <= 10 && n_true>5",51,0.8,1.2,"Momentum resolution","# particles");
    compareTH1D resolution10_15("reco_e/true_e","is_true && is_reco && n_true <= 15 && n_true>10",51,0.8,1.2,"Momentum resolution","# particles");

    resolution1_5.DrawAxes();

    resolution1_5.setOCLineColourAndStyle(-1,3);
    resolution1_5.setClassicLineColourAndStyle(-1,3);
    resolution5_10.setOCLineColourAndStyle(-1,2);
    resolution5_10.setClassicLineColourAndStyle(-1,2);
    resolution1_5.Draw("same","PF");
    resolution5_10.Draw("same","PF");
    resolution10_15.Draw("same","PF");

    TLegend * leg_c = new TLegend(0.6,0.6, 0.85, 0.85);
    leg_c -> SetLineWidth(1);
    makeLegEntry(leg_c,"1-5 particles","l",kBlack,3);
    makeLegEntry(leg_c,"6-10 particles","l",kBlack,2);
    makeLegEntry(leg_c,"10-15 particles","l",kBlack,1);
    leg_c->Draw("same");

    cv->Print("resolution_pf.pdf");

    resolution1_5.DrawAxes();
    resolution1_5.Draw("same","OC");
    resolution5_10.Draw("same","OC");
    resolution10_15.Draw("same","OC");
    leg_c->Draw("same");

    cv->Print("resolution_oc.pdf");

    ///////////////////////////////////////////


    compareProfile variance1_5("(reco_e/true_e - 1)**2:true_e",  "abs(reco_e/true_e - 1)<0.1 && is_true && is_reco && n_true <= 5",10,0,200,"p(t) [GeV]","Variance");
    compareProfile variance5_10("(reco_e/true_e - 1)**2:true_e", "abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 10 && n_true>5",10,0,200,"True momentum [GeV]","Variance");
    compareProfile variance10_15("(reco_e/true_e - 1)**2:true_e","abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 15 && n_true>10",10,0,200,"True momentum [GeV]","Variance");

    variance1_5.setOCLineColourAndStyle(-1,3);
    variance1_5.setClassicLineColourAndStyle(-1,3);
    variance5_10.setOCLineColourAndStyle(-1,2);
    variance5_10.setClassicLineColourAndStyle(-1,2);

    variance1_5.DrawAxes();
    variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,0.003);
    variance1_5.Draw("same","PF");
    variance5_10.Draw("same","PF");
    variance10_15.Draw("same","PF");

    cv->Print("variance_pf.pdf");

    variance1_5.DrawAxes();
    variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,0.003);
    variance1_5.Draw("same","OC");
    variance5_10.Draw("same","OC");
    variance10_15.Draw("same","OC");

    cv->Print("variance_oc.pdf");


    variance1_5.DrawAxes();
    variance1_5.Draw("same","");
    variance5_10.Draw("same","");
    variance10_15.Draw("same","");

    placeLegend(leg_a2, 0.2, 0.55);
    leg_a2->Draw("same");

    cv->Print("variance.pdf");

    compareTH1D outside_var1_5("true_e","1./1000.*(abs(reco_e/true_e - 1)>0.1 && is_true && is_reco && n_true <= 5)",10,0,200,"True momentum [GeV]","Fraction with |p(r)/p(t) - 1| > 0.1 [%]");
    compareTH1D outside_var5_10("true_e","1./1000.*(abs(reco_e/true_e - 1)>0.1 &&is_true && is_reco && n_true <= 10 && n_true>5)",10,0,200,"True momentum [GeV]","Misidentified fraction [%]");
    compareTH1D outside_var10_15("true_e","1./1000.*(abs(reco_e/true_e - 1)>0.1 &&is_true && is_reco && n_true <= 15 && n_true>10)",10,0,200,"True momentum [GeV]","Misidentified fraction [%]");

    outside_var1_5.setOCLineColourAndStyle(-1,3);
    outside_var1_5.setClassicLineColourAndStyle(-1,3);
    outside_var5_10.setOCLineColourAndStyle(-1,2);
    outside_var5_10.setClassicLineColourAndStyle(-1,2);

    outside_var1_5.DrawAxes();
    outside_var1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,2.6);
    outside_var1_5.Draw("same","");
    outside_var5_10.Draw("same","");
    outside_var10_15.Draw("same","");

    placeLegend(leg_a2, 0.6, 0.55);
    leg_a2->Draw("same");

    cv->Print("misidentified.pdf");

    return 0;
}


int main(int argc, char* argv[]){
    return plotscript(argc,argv);
}
