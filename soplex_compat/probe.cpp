// Candidate-only exact I/O probe. No native status is a proof acceptance gate.
#include "soplex.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace soplex;

static void require(bool ok) { if(!ok) throw std::runtime_error("SoPlex API rejected operation"); }

static void dump(SoPlex& lp, const NameSet& rows, const NameSet& cols,
                 const std::string& path) {
    std::ofstream f(path);
    f << "SOPLEX_RATIONAL_READBACK_V1\n";
    f << "COLUMNS " << lp.numColsRational() << "\n";
    for(int j=0;j<lp.numColsRational();++j)
        f << cols[j] << " " << lp.objRational(j) << " "
          << lp.lowerRational(j) << " " << lp.upperRational(j) << "\n";
    f << "ROWS " << lp.numRowsRational() << "\n";
    for(int i=0;i<lp.numRowsRational();++i) {
        const auto& row=lp.rowVectorRational(i);
        f << rows[i] << " ";
        if(lp.lhsRational(i)==-Rational(lp.realParam(SoPlex::INFTY))) f << "-inf";
        else f << lp.lhsRational(i);
        f << " " << lp.rhsRational(i) << " " << row.size();
        for(int k=0;k<row.size();++k) f << " " << cols[row.index(k)] << " " << row.value(k);
        f << "\n";
    }
    f << "END\n"; f.flush(); require(bool(f));
}

int main(int argc,char**argv) {
    if(argc!=3) return 2;
    try {
        SoPlex lp;
        require(lp.setIntParam(SoPlex::VERBOSITY,0));
        require(lp.setIntParam(SoPlex::READMODE,1));
        require(lp.setIntParam(SoPlex::SYNCMODE,1));
        require(lp.setIntParam(SoPlex::SOLVEMODE,2));
        require(lp.setIntParam(SoPlex::CHECKMODE,2));
        require(lp.setIntParam(SoPlex::OBJSENSE,SoPlex::OBJSENSE_MINIMIZE));
        require(lp.setRealParam(SoPlex::FEASTOL,0));
        require(lp.setRealParam(SoPlex::OPTTOL,0));
        require(lp.setRealParam(SoPlex::TIMELIMIT,10));
        NameSet rows,cols;
        require(lp.readFile(argv[1],&rows,&cols));
        const std::string prefix(argv[2]);
        dump(lp,rows,cols,prefix+".before");
        std::ofstream settings(prefix+".settings");
        settings << lp.intParam(SoPlex::READMODE) << " " << lp.intParam(SoPlex::SYNCMODE)
          << " " << lp.intParam(SoPlex::SOLVEMODE) << " " << lp.intParam(SoPlex::CHECKMODE)
          << " " << lp.intParam(SoPlex::OBJSENSE) << " " << lp.realParam(SoPlex::FEASTOL)
          << " " << lp.realParam(SoPlex::OPTTOL) << " " << lp.realParam(SoPlex::TIMELIMIT) << "\n";
        settings.flush(); require(bool(settings));
        auto status=lp.optimize();
        dump(lp,rows,cols,prefix+".after");
        std::ofstream f(prefix+".point");
        f << "SOPLEX_CANDIDATE_V1 " << int(status) << "\n";
        VectorRational x(lp.numColsRational());
        if(lp.isPrimalFeasible() && lp.getPrimalRational(x)) {
            f << "POINT " << lp.numColsRational() << "\n";
            for(int j=0;j<lp.numColsRational();++j) f << cols[j] << " " << x[j] << "\n";
        } else f << "NO_POINT\n";
        f << "END\n"; f.flush(); require(bool(f));
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << "\n"; return 2; }
}
