#include<bits/stdc++.h>
#define debug(...) fprintf(stderr, __VA_ARGS__), fflush(stderr)
#define time__(d) for ( auto blockTime = make_pair(chrono::high_resolution_clock::now(), true);blockTime.second;\
        debug("%s: %lld ms\n", d, chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - blockTime.first).count()), blockTime.second = false \
    )
const long int BATCH = 2;
using namespace std;

class Activations
{
	public:

		long double Tanh(long double x)
		{
			return (1-exp(-2*x))/(1+exp(-2*x));
		}

		long double Leakyrelu(long double x,long double alpha)
		{
			return (x<0)?(alpha*x):x;
		}

		long double Sigmoid(long double x)
		{
			return 1/(1+exp(-x));
		}
};

class Generator
{
	public:
		
		long int num_layers;  //excludes input layer
		long int inp_dim; // number of dimensions in input
		vector<vector<long double> > input; 
		vector<long int> layersizes; //sizes of hidden layers and output layer

		Generator(long int inp_dim, vector<vector<long double> >input, long int num_layers, vector<long int> layersizes)
		{
			Generator::inp_dim = inp_dim;
			Generator::input = input;
			Generator::num_layers = num_layers;
			Generator::layersizes = layersizes;
		}

		~Generator()
		{
			cout<<"Generator ends here..";
		}

		void batchNormOnAct(vector<vector<long double> > &samplevec, long double epsilon,long double gamma, long double beta)
		{
			for(long int d = 0; d < samplevec[0].size(); d++) //d is the number of nodes in current layer on which batchnorm is being performed
			{
				long double mean = 0.0;
				for(long int m = 0; m < BATCH ; m++)
				{
					mean+=samplevec[m][d];
				}
				mean/=BATCH;

				long double variance = 0.0;
				for(long int m = 0; m < BATCH ; m++)
				{
					variance += pow(samplevec[m][d] - mean,2);
				}

				variance/=BATCH;

				//normalizing finally
				for(long int m = 0; m < BATCH ; m++)
				{
					samplevec[m][d] = (samplevec[m][d] - mean)/(sqrt(variance + epsilon));
					samplevec[m][d] = gamma*samplevec[m][d] + beta;
				}
			}
		}

		void leakyReluOnAct(vector<vector<long double> > &samplevec, long double alpha)
		{
			Activations a;

			for(long int d = 0; d < samplevec[0].size(); d++)
			{
				for(long int m = 0; m < BATCH ; m++)
				{
					samplevec[m][d] = a.Leakyrelu(samplevec[m][d], alpha);
				}
			}
		}

		void tanh(vector<vector<long double> > &samplevec)
		{
			Activations a;

			for(long int d = 0; d < samplevec[0].size(); d++)
			{
				for(long int m = 0; m < BATCH ; m++)
				{
					samplevec[m][d] = a.Tanh(samplevec[m][d]);
				}
			}
		}

		vector<vector<long double> > forward()
		{
			vector<vector<long double> > curr_layerinp = input;

			for(long int l = 0; l<num_layers; l++)
			{
				vector<vector<long double> > next_layerinp(BATCH);
				vector<vector<long double> > weights(layersizes[l+1]);

				for(long int n=0; n<layersizes[l+1];n++)
				{
					for(long int w = 0; w < layersizes[l] ; w++)
					{
						long double num;
						cin>>num;
						weights[n].push_back(num);
					}
				}

				for(long int m = 0; m < BATCH; m++)
				{
					for(long int n = 0; n < layersizes[l+1] ; n++)
					{
						//calculate linear sum
						long double linearsum = 0.0;
						for(long int w = 0; w < layersizes[l] ; w++)
						{
							linearsum+=(weights[n][w]*curr_layerinp[m][w]); 
						}

						// no bias for now
						// long double bias;
						// cin>>bias;
						// linearsum+=bias;

						next_layerinp[m].push_back(linearsum); //making the node value equal to the linear sum + bias
					}
					cout<<"Batch "<<m<<" is done.\n";
				}

				if(l>0 && l < num_layers-1) //batchnorm and leakyrelu on hidden layers only
				{
					batchNormOnAct(next_layerinp, 0.8, 1, 0);
					leakyReluOnAct(next_layerinp,0.2);
				}

				if(l==num_layers - 1)
				{
					tanh(next_layerinp);
				}

				curr_layerinp.clear();
				curr_layerinp = next_layerinp;
				next_layerinp.clear();

				cout<<"Layer "<<l<<" is done.\n";
			}	

			return curr_layerinp;
		}

		void printinput(vector<vector<long double> > v)
		{
			for(long int i=0;i<v.size();i++)
			{
				for(long int j = 0; j < v[i].size(); j++)
				{
					cout<<v[i][j]<<' ';
				}
				cout<<'\n';
			}
		}

		void printLi1D(vector<long int> v)
		{
			for(long int i=0;i<v.size();i++)
			{
				cout<<v[i]<<' ';
			}
			cout<<'\n';
		}
};
int main()
{	
	freopen("output.txt","w",stdout);

	//number of layers, layer sizes and weight matrices are given in text file input.txt
	cout<<setprecision(10);

	freopen("input.txt","r",stdin);

	long int inp_dim;
	cin>>inp_dim;

	vector<vector<long double> > input(BATCH); //double because values are from [0,1]
	for(long int b = 0; b < BATCH ; b++)
	{
		for(long int i = 0 ; i < inp_dim ;i++)
		{
			long double num;
			cin>>num;
			input[b].push_back(num);
		}
	}

	fclose(stdin);

	freopen("weights.txt","r",stdin);

	long int num_layers;

	cin>>num_layers;

	vector<long int> layersizes; //layer 0 to output layer
	
	layersizes.push_back(inp_dim);
	
	//L+1 layers with 1 input layer, L-1 hidden layers and 1 output layer.
	for(long int i=0;i<num_layers;i++)
	{
		long int temp;
		cin>>temp;
		layersizes.push_back(temp);
	}
	
	Generator g(inp_dim,input,num_layers,layersizes);

	time__("forward")
	{
		vector<vector<long double> > res = g.forward();
		g.printinput(res);
	}

	fclose(stdin);
	return 0;
}