function [Est_QE,AsyV_QE,Wald,Stability,Mean,U,DQ,etahat,q]=GEWMA_Quantile(x,tau,r,k,index)

%Inputs:
%   Data:                                        x     (note that x must be a row vector)
%   Quantile level:                              tau   (the value of tau lies between 0 and 1)
%   Indicator for the preliminary estimator:     r     (e.g., r=2 is corresponding to the Guassian QMLE)
%   Lag:                                         k     (the value of lag for the DQ test)
%   Exponent:                                    index (the value of exponent for the DQ test)
%   

%Outputs:
%   Weighted Quantile estimator (QE):                               Est_QE 
%   Estimated standard errors of weighted QE:                       AsyV_QE 
%   The value of Wald test for linear constraint and its p-value:   Wald
%   The value of Stability test and its p-value:                    Stability
%   The value of Mean Invariance test and its p-value:              Mean
%   The value of Unit-root test and its p-value:                    U
%   The value of DQ test and its p-value:                           DQ
%   The model residuals:                                            etahat
%   The estimated VaR sequence:                                     q


y=log(x(2:end))-log(x(1:end-1));  % Log-differencing
y0=y(1);  % y(1) is non-zero
y=y(2:end);

n=length(y);
thetahat=[0.1 0.8]; %Initial point

options=optimset('Display','off','LargeScale','off','GradObj','off',...
'TolX',1e-8,'TolCon',1e-8,'MaxIter',1000,'MaxFunEvals',1000, 'Algorithm','interior-point'); % Using interior-point algorithm

AStar=[-1 0; 0 -1];
AboundStar=[-0.0000000001;-0.0000000001];


Est_QE=zeros(1,2);
AsyV_QE=zeros(1,2);

T_w=zeros(1,1);
T_w_pvalue=zeros(1,1);

T_gamma_s=zeros(1,1);
T_gamma_s_pvalue=zeros(1,1);

T_gamma_m=zeros(1,1);
T_gamma_m_pvalue=zeros(1,1);

T_DQ=zeros(1,1);
T_DQ_pvalue=zeros(1,1);


%%%%%%%%%%%%%%%%%%%%%% Estimation
QMLE=@(theta)GQMLE_estimation(y,y0,theta,r);

theta_qmle=fmincon(QMLE,thetahat,AStar,AboundStar,[],[],[],[],[],options);  %% The preliminary estimator GQMLE
       
qstar=weight_fun(y,y0,theta_qmle);
       
QE=@(theta)Weighted_Quantile(y,y0,theta,qstar,tau); 
    
theta_quantile=fmincon(QE,thetahat,AStar,AboundStar,[],[],[],[],[],options); %% The weighted quantile estimator
    
[etahat,q]=residual(y,y0,theta_quantile);
   
Sigma_QE=Variance_Sig_QE(theta_quantile,etahat);

band=1.06*n^(-1/5)*std(etahat);
band=band*(1+(35/48)*kurtosis(etahat)+(35/32)*(skewness(etahat))^2+(385/1024)*(kurtosis(etahat))^2)^(-1/5);
f1=sum(normpdf((-1-etahat)./band))/(n*band);   %%%% f(-1)
tau_n=(tau-tau^2)/f1^2;

Est_QE(1,1:2)=theta_quantile;
std_QE=(diag(tau_n*inv(Sigma_QE))).^(1/2)/sqrt(n);
AsyV_QE(1,1:2)=std_QE;   %% The estimated standard errors of weighted quantile estimator
    
    
%%%%%%%%%%%%%%%%%%%%%% linear constraint test
A_matrix=[1 1];
c_matrix=1;
W_test=n*((A_matrix*theta_quantile'-c_matrix)')*inv(A_matrix*tau_n*inv(Sigma_QE)*(A_matrix'))...
             *(A_matrix*theta_quantile'-c_matrix);
T_w(1,1)= W_test;
T_w_pvalue(1,1)= 1-chi2cdf(W_test,1);
Wald=[T_w(1,1) T_w_pvalue(1,1)];
       
%%%%%%%%%%%%%%%%%%%%%% stablility test 
gamma_s=mean(log(theta_quantile(2)+theta_quantile(1)*abs(etahat)));
gamma_s_var=var(log(theta_quantile(2)+theta_quantile(1)*abs(etahat)));
T_gamma_s(1,1)=sqrt(n)*gamma_s/sqrt(gamma_s_var);
T_gamma_s_pvalue(1,1)= 2*normcdf(-T_gamma_s(1,1));
Stability=[T_gamma_s(1,1) T_gamma_s_pvalue(1,1)];
    
%%%%%%%%%%%%%%%%%%%%%%% mean test
v1=mean(theta_quantile(2)./(theta_quantile(2)+theta_quantile(1)*(abs(etahat))));
omega_bar=mean((tau-(etahat<-1)).*abs(etahat));
Lambda=[0 1 theta_quantile(1)];
Lambda(1,2)=1-theta_quantile(1)*v1*mean(abs(etahat))/(theta_quantile(2)*(1-v1));
       
D_theta=[1/theta_quantile(1); v1/(theta_quantile(2)*(1-v1))];
Sigma_m=zeros(3,3);
Sigma_m(1:2,1:2)=tau_n*inv(Sigma_QE);
Sigma_m(1:2,3:3)=-(omega_bar/f1)*inv(Sigma_QE)*D_theta;  
Sigma_m(3:3,1:2)=Sigma_m(1:2,3:3)';
Sigma_m(3:3,3:3)=var(abs(etahat));
       
sigma_all=Lambda*Sigma_m*(Lambda');
gamma_m=theta_quantile(2)+theta_quantile(1)*mean(abs(etahat));
    
T_gamma_m(1,1)=sqrt(n)*(gamma_m-1)/sqrt(sigma_all);
T_gamma_m_pvalue(1,1)= 2*normcdf(-T_gamma_m(1,1));

Mean=[T_gamma_m(1,1) T_gamma_m_pvalue(1,1)];
    
%%%%%%%%%%%%%%%%%%%%%% DQ test
lambda_exp=zeros(1,n);
for t=1:n
    lambda_exp(t)=theta_quantile(2)^(t-1);
end
abs_y=abs(y);
rev_abs_y=[abs_y(end-1:-1:1) abs(y0)];
    
Q=zeros(1,n);
QP=zeros(n,2);  %%%% QP=Q/partial_Q
QP(1,1)=1/theta_quantile(1);
for t=2:n
    Q(t)=theta_quantile(1)*sum(lambda_exp(1:t).*rev_abs_y(end-t+1:1:end));
    QP(t,1)=1/theta_quantile(1);
    QP(t,2)=sum((lambda_exp(1:t-1).*rev_abs_y(end-t+2:1:end)).*[1:1:t-1])...
              /sum(lambda_exp(1:t).*rev_abs_y(end-t+1:1:end));
end
    
Hit=(y+q<0)-tau;

Upsilon=zeros(k,k);
Gamma=zeros(k,2);
J=zeros(k,1);
for t=k+1:n   
    X_e=[abs(etahat(t-1:-1:t-k)).^(index)]';
    Upsilon=Upsilon+X_e*(X_e')/n;
    Gamma=Gamma+X_e*QP(t,:)/n;
    J=J+Hit(t)*X_e/sqrt(n);
end

DQ=(J')*inv(Upsilon-Gamma*inv(Sigma_QE)*(Gamma'))*J/(tau-tau^2);  
T_DQ(1,1)=DQ;
T_DQ_pvalue(1,1)= 1-chi2cdf(T_DQ(1,1),k);
DQ=[T_DQ(1,1) T_DQ_pvalue(1,1)];     

%%%%%%%%%%%%%%%% Testing zero-drift
[U_test,U_test_pvalue]=testing_omega_zero(q);
U=[U_test U_test_pvalue];
     
end


function S=GQMLE_estimation(y,y0,theta,r)

S=0;
n=length(y);
q=0.01*ones(1,n);

if r>0

q(1)=theta(1)*abs(y0);
S=S+r*log(q(1))+(abs(y(1))/q(1))^(r);

for i=2:n
    
    q(i)=theta(1)*abs(y(i-1))+theta(2)*q(i-1);
    S=S+r*log(q(i))+(abs(y(i))/q(i))^(r);
    
end

end

if r==0
    
q(1)=theta(1)*abs(y0);
S=S+(log(abs(y(1)))-log(q(1)))^2;

for i=2:n
    
    q(i)=theta(1)*abs(y(i-1))+theta(2)*q(i-1);
    
    if y(i)~=0
    
    S=S+(log(abs(y(i)))-log(q(i)))^2;
    
    end
    
end

end

end

function q=weight_fun(y,y0,theta_qmle)


n=length(y);

q=0.01*ones(1,n);

theta=theta_qmle;

q(1)=theta(1)*abs(y0);

for i=2:n
    
    q(i)=theta(1)*abs(y(i-1))+theta(2)*q(i-1);
    
end

end

function S=Weighted_Quantile(y,y0,theta,qstar,tau)

S=0;

n=length(y);

q=0.01*ones(1,n);

q(1)=theta(1)*abs(y0);

S=S+((y(1)+q(1))/qstar(1))*(tau-(y(1)<-q(1)));

for i=2:n
    
    q(i)=theta(1)*abs(y(i-1))+theta(2)*q(i-1);
    S=S+((y(i)+q(i))/qstar(i))*(tau-(y(i)<-q(i)));
    
end

end

function [etahat,q]=residual(y,y0,theta)

n=length(y);

etahat=0.01*ones(1,n);
q=0.01*ones(1,n);

q(1)=theta(1)*abs(y0);
etahat(1)=y(1)/q(1);

for i=2:n
    
    q(i)=theta(1)*abs(y(i-1))+theta(2)*q(i-1);
    etahat(i)=y(i)/q(i);
    
end

end

function Sigma=Variance_Sig_QE(theta,etahat)

       Sigma=zeros(2,2);
    
       v1=mean(theta(2)./(theta(2)+theta(1)*(abs(etahat))));
       v2=mean((theta(2)./(theta(2)+theta(1)*(abs(etahat)))).^2);
    
       Sigma(1,1)=1/(theta(1)^2);
       Sigma(1,2)=v1/(theta(1)*theta(2)*(1-v1));
       Sigma(2,1)=Sigma(1,2);
       Sigma(2,2)=(1+v1)*v2/((theta(2)^2)*(1-v1)*(1-v2));

end

function [stat1,p_value]=testing_omega_zero(h)

z_star = log(h);
n=length(z_star);

b_star_hat=sum(z_star(2:end).*z_star(1:end-1))/(sum(z_star(1:end-1).^2));

u_error = zeros(1,n-1);
u_error(1:end) = z_star(2:end) - b_star_hat*z_star(1:end-1);
se_u_error = sqrt(mean(u_error.^2)/((n-1)*mean(z_star(2:end).^2)));
stat1 = ((b_star_hat-1)/se_u_error)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% calculating p-value

n=10000;
rep=10000;
T=zeros(1,rep);
Count=0;

for i=1:rep

eta=randn(1,n);
cum_eta=cumsum(eta)/sqrt(n);
cum_eta_star=[0 cum_eta(1:end-1)];

D1=(cum_eta(end)^2-1)/2;
D2=sum(cum_eta_star.^2)/n;

T(i)=(D1/sqrt(D2))^2;

if T(i)>stat1
    
    Count=Count+1;
    
end

end

p_value=Count/rep;

end
