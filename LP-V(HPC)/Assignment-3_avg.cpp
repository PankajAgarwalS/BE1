#include<iostream>
#include<omp.h>
using namespace std;

int main()
{
   int a[100],n,i;
   cout<<"Enter the number of elements in array : ";
   cin>>n; 
   cout<<"\n Enter array elements : ";
   for(i=0;i<n;i++)
   {
      cin>>a[i];
   }
   cout<<"\n Array elements are: ";
   for(i=0;i<n;i++)
   {
      cout<<a[i]<<"\t";
   }
   float avg=0,sum=0;
   #pragma omp parallel 
   {
     int id=omp_get_thread_num();
     #pragma omp for
     for(i=0;i<n;i++)
     {
       sum=sum+a[i];
       cout<<"\nFor i= " <<i<<" thread "<<id<<" is executing "<<endl;
     }
   }
   avg=sum/n;
   cout<<"Output = "<<avg<<endl;
}
