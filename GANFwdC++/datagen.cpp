#include<bits/stdc++.h>
using namespace std;

void getmean(vector<vector<long double> >v)
{
	cout<<v[0][1]<<endl;
}
int main()
{
	freopen("input.txt","w",stdout);
	srand(time(NULL));

	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);

	cout<<100<<endl;

	for(int i=0;i<100;i++)
	{
		double num = distribution(generator);
		if(num>1)
			num--;
		else if(num<-1)
			num++;
		cout<<num<<' ';
	}

	cout<<'\n';

	for(int i=0;i<100;i++)
	{
		double num = distribution(generator);
		if(num>1)
			num--;
		else if(num<-1)
			num++;
		cout<<num<<' ';
	}

	fclose(stdout);

	freopen("weights.txt","w",stdout);

	cout<<4<<endl;
	cout<<256<<' '<<512<<' '<<1024<<' '<<784<<endl;

	int arr[5] = {100,256,512,1024,784};
	// int arr[5] = {1,2,3,4,5};
	for(int i=0;i<sizeof(arr)/sizeof(arr[0]);i++)
	{
		for(int j=0;j<arr[i+1];j++)
		{
			for(int k=0;k<arr[i];k++)
			{
				cout<<distribution(generator)<<' ';
			}
			cout<<'\n';
		}
	}


	fclose(stdout);

	return 0;

}