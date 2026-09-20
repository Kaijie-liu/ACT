// Import/readback ONLY: deliberately no optimize(), basis or candidate API.
#include "soplex.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace soplex;
static void need(bool ok) { if(!ok) throw std::runtime_error("rational import failure"); }
int main(int argc,char**argv) {
    if(argc!=3) return 2;
    try {
        SoPlex lp;
        need(lp.setIntParam(SoPlex::VERBOSITY,0));
        need(lp.setIntParam(SoPlex::READMODE,1));
        need(lp.setIntParam(SoPlex::SYNCMODE,1));
        NameSet rows,cols;
        need(lp.readFile(argv[1],&rows,&cols));
        need(lp.intParam(SoPlex::READMODE)==1 && lp.intParam(SoPlex::SYNCMODE)==1);
        need(lp.intParam(SoPlex::OBJSENSE)==SoPlex::OBJSENSE_MINIMIZE);
        need(lp.realParam(SoPlex::OBJ_OFFSET)==0);
        std::ofstream f(argv[2]);
        f << "SOPLEX_RATIONAL_READBACK_V1\nCOLUMNS " << lp.numColsRational() << "\n";
        for(int j=0;j<lp.numColsRational();++j)
            f << cols[j] << " " << lp.objRational(j) << " " << lp.lowerRational(j)
              << " " << lp.upperRational(j) << "\n";
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
        f << "END\n";f.flush();need(bool(f));return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << "\n";return 2; }
}
