/* file : gfg.c */
/*
def in_or_out(self,refx,refy, meg):
        shx = meg.Plcoords[:,0]
        shy = meg.Plcoords[:,1]
        #step 1: eliminate stuff outside the bounding box
        if(refx<min(shx) or refx>max(shx) or refy>max(shy) or refy<min(shy)): return(0)
        #step 2: ray tracing horizontal
        shy = np.append(shy,shy[0])
        shx = np.append(shx,shx[0])
        intsecty = (np.asarray([(shy[i]-refy)*(shy[i+1]-refy) if(shx[i]>refx) else 0 
            for i in range(0,len(shy)-1)])<0).sum()
        if(intsecty%2 !=0): return(1)
        else: return(0)*/

//swig -c++ -python gfg.i
//c++ -c -fpic gfg_wrap.cxx gfg.cpp -I/usr/include/python3.8
//c++ -shared gfg.o gfg_wrap.o -o _gfg.so

#include<iostream>
#include<vector>

void printed(std::vector<float> yay) {
    for(int i=0;i<yay.size();i++) std::cout<<yay[i]<<" ";
}

void printed(std::vector<int> yay) {
    for(int i=0;i<yay.size();i++) std::cout<<yay[i]<<" ";
}

float max(std::vector<float> yay) {
    float m=0;
    for(int i=0;i<yay.size();i++) {
        if (yay[i]>m) m=yay[i];
    }
    return(m);
}

float min(std::vector<float> yay) {
    float m=0;
    for(int i=0;i<yay.size();i++) {
        if (yay[i]<m) m=yay[i];
    }
    return(m);
}

int hello() {
    return(100);
}

//void printed(vector<float> yay);

std::vector<int> in_or_out(std::vector<float> refx,std::vector<float> refy, std::vector<float> newshx , std::vector<float> newshy) {

    //eliminate stuff outside bounding boxes
    int n=newshx.size();
    float minshx=min(newshx);
    float maxshx=max(newshx);
    float minshy=min(newshy);
    float maxshy=max(newshy);

    std::vector<int> distarr;


    for(int j=0;j<refx.size(); j++) {
        if(refx[j]<minshx || refx[j]>maxshx || refy[j]>maxshy || refy[j]<minshy) distarr.push_back(0);

        newshx.push_back(newshx[0]);
        newshy.push_back(newshy[0]);

        std::vector<float> temp;

        for(int i=0; i<n; i++) {
            if(newshx[i]> refx[j]) {temp.push_back((newshy[i]-refy[j])*(newshy[i+1]-refy[j]) < 0);}
            else {temp.push_back(0);}
        }

        int intsecty = 0;
        for(int i=0; i<temp.size();i++) {intsecty+=temp[i]; }

        if(intsecty%2 !=0) distarr.push_back(1);
        else distarr.push_back(0); 
    }

    return(distarr);

}
  
/*
int main() {
    std::cout<<hello()<<std::endl;

    float shx[] = {-2.0,2.0,2.0,-2.0};
    float shy[] = {-2.0,-2.0,2.0,2.0};

    float refx[] = {1,2,1,3,4,0};
    float refy[]= {2,3,1,-2,1,0};
    std::vector<int> val=in_or_out(refx,refy, shx, shy, 4,6);

    printed(val);

    return(0);
}*/
