

#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2.h"
#include "TTree.h"
#include <iostream>
#include "TEfficiency.h"
#include "TStyle.h"
#include "globals.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TProfile.h"

template<class T>
class comparePlotWithAxes {
public:
    comparePlotWithAxes(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin,
            TString xaxis, TString yaxis){
        axes_=0;
        classic_o_=0;
        oc_o_=0;
        objstr_="";
        objstr_+=counter;
        counter++;
        createAxes(nbins,minbin,maxbin,xaxis,yaxis);
    }
    virtual ~comparePlotWithAxes(){
        delete classic_o_;
        delete oc_o_;
    }

    void setClassicLineColourAndStyle(int col,int style=-1000){
        if(col>=0)
            (classic_o_)->SetLineColor(col);
        if(style>-1000)
            (classic_o_)->SetLineStyle(style);
    }
    void setOCLineColourAndStyle(int col,int style=-1000){
        if(col>=0)
            (oc_o_)->SetLineColor(col);
        if(style>-1000)
            (oc_o_)->SetLineStyle(style);
    }

    T* getOC(){return oc_o_;}
    T* getCl(){return classic_o_;}

    virtual void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin)=0;//{}

    void Draw(TString opt="", TString which=""){//always draws with same
        if(which.Contains("OC")){
            (oc_o_)->Draw(opt);
        }
        if(which.Contains("PF")){
            (classic_o_)->Draw(opt);
        }
        if(which.Length()<1){
            (classic_o_)->Draw(opt);
            (oc_o_)->Draw(opt);
        }

    }
    void DrawAxes(){
        axes_->Draw("AXIS");
    }

    void setStyleDefaults(){
        (classic_o_)->SetLineColor(global::defaultClassicColour);
        (classic_o_)->SetLineWidth(2);
        (oc_o_)->SetLineColor(global::defaultOCColour);
        (oc_o_)->SetLineWidth(2);
    }


    TH1D* AxisHisto(){return axes_;}

private:

    static int counter;
protected:

    T* classic_o_;
    T* oc_o_;
    TH1D* axes_;
    void createAxes(int nbins, double minbin, double maxbin,TString xaxis, TString yaxis){
        if (axes_)
            delete axes_;
        axes_ = new TH1D("axis"+objstr_,"axis"+objstr_,nbins,minbin,maxbin);

        axes_->GetXaxis()->SetTitle(xaxis);
        axes_->GetYaxis()->SetTitle(yaxis);
        axes_->GetYaxis()->SetTitleOffset(1.45);

        axes_->GetXaxis()->SetLabelSize(0.05);
        axes_->GetYaxis()->SetLabelSize(0.05);
        axes_->GetXaxis()->SetTitleSize(0.05);
        axes_->GetYaxis()->SetTitleSize(0.05);

    }
    TString objstr_;
    std::vector<TObject*> otherobj_;

};
template<class T>
int comparePlotWithAxes<T>::counter=0;


class compareEfficiency: public comparePlotWithAxes<TEfficiency> {
public:
    compareEfficiency(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin, TString xaxis, TString yaxis):
        comparePlotWithAxes(var,selection,selectionpass,nbins,minbin,maxbin,xaxis,yaxis){
        createObj(var,selection,selectionpass,nbins,minbin,maxbin);
        setStyleDefaults();
    }

    void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_ = makeEfficiency(global::classic_tree, "cl"+objstr_,var, selection,selectionpass,nbins,minbin,maxbin);
        oc_o_ = makeEfficiency(global::oc_tree, "oc"+objstr_,var, selection,selectionpass,nbins,minbin,maxbin);

    }

private:
    TEfficiency * makeEfficiency(TTree* t, TString add, TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin){

        TH1D * hpass  = new TH1D("hpass"+add,"hpass"+add, nbins, minbin, maxbin);
        otherobj_.push_back(hpass);
        TH1D * htotal = new TH1D("htotal"+add,"htotal"+add, nbins, minbin, maxbin);
        otherobj_.push_back(htotal);

        TEfficiency* eff = new TEfficiency("eff"+add,"eff"+add,nbins,minbin,maxbin);
        eff->SetUseWeightedEvents(true);
        eff->SetStatisticOption(TEfficiency::kFNormal);

        std::cout << var+">>"+"htotal"+add << std::endl;

        t->Draw(var+">>"+"htotal"+add,selection);
        t->Draw(var+">>"+"hpass"+add,selectionpass);

        eff->SetTotalHistogram(*htotal,"");
        eff->SetPassedHistogram(*hpass,"");

        return eff;
    }

};

class compareTH1D: public comparePlotWithAxes<TH1D> {
public:
    compareTH1D(TString var, TString selection, int nbins, double minbin, double maxbin, TString xaxis, TString yaxis):
        comparePlotWithAxes(var,selection,"",nbins,minbin,maxbin,xaxis,yaxis){
        createObj(var,selection,"",nbins,minbin,maxbin);
        setStyleDefaults();
    }

    void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_  = new TH1D("ha"+objstr_,"ha"+objstr_, nbins, minbin, maxbin);
        oc_o_  = new TH1D("hb"+objstr_,"hb"+objstr_, nbins, minbin, maxbin);

        global::classic_tree->Draw(var+">>"+"ha"+objstr_,selection);
        global::oc_tree->Draw(var+">>"+"hb"+objstr_,selection);

        mergeOverflows(classic_o_);
        mergeOverflows(oc_o_);

        auto max=oc_o_->GetMaximum();

        AxisHisto()->GetYaxis()->SetRangeUser(0, max*1.1);

    }
private:
    void mergeOverflows(TH1D* h){
        auto UF = h->GetBinContent(0);
        h->SetBinContent(1,UF+h->GetBinContent(1));
        auto OF = h->GetBinContent(h->GetNbinsX()+1);
        h->SetBinContent(1,OF+h->GetBinContent(h->GetNbinsX()));
    }
};



class compareProfile: public comparePlotWithAxes<TProfile> {
public:
    compareProfile(TString var, TString selection, int nbins, double minbin, double maxbin, TString xaxis, TString yaxis):
        comparePlotWithAxes(var,selection,"",nbins,minbin,maxbin,xaxis,yaxis){
        createObj(var,selection,"",nbins,minbin,maxbin);
        setStyleDefaults();
    }

    void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_  = new TProfile("ha"+objstr_,"ha"+objstr_, nbins, minbin, maxbin);
        oc_o_       = new TProfile("hb"+objstr_,"hb"+objstr_, nbins, minbin, maxbin);

        global::classic_tree->Draw(var+">>"+"ha"+objstr_,selection,"prof");
        global::oc_tree->Draw(var+">>"+"hb"+objstr_,selection,"prof");

        auto max=oc_o_->GetMaximum();

        AxisHisto()->GetYaxis()->SetRangeUser(0, max*1.1);

    }
};



TLegendEntry * makeLegEntry(TLegend* leg, TString name, TString option, int col, int style=-1);


void placeLegend(TLegend* leg, double x1, double y1, double x2=-1, double y2=-1);






