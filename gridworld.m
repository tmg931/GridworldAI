%Gridworld
clc;
m = 7; %length of gridworld
n = 7; %height of gridworld
N = m*n; %number of states
NA = 4; %number of actions
p = .05; % living penalty
noise = 0.2; %probability of going the wrong way
gamma = 0.9; %discount rate
rate = 1; %learning rate
epsilon = 1; %probability of taking 2nd or 3rd best action in exploration phase
QValues = zeros(49,5); %Q Values
totalscore = 0; %counter for total score
post = 0; %test counter for +100 reward trials



%Transitions - this array shows where each action will take the agent 
T = zeros(N,NA+1);
T(:,1)=1:m*n;
T(:,2)=T(:,1)-m; % move one space up
T(:,3)=T(:,1)+1; % move one space right
T(:,4)=T(:,1)+m; % move one space down
T(:,5)=T(:,1)-1; % move one space left

%Uncomment to reset Q values 
%Q = zeros(m*n,NA+1); %q-vaules for each action

%handling edge cases
%top row
T(1:m,2) = T(1:m,1); %don't move if up is selected

%bottom row
T((m*(n-1)+1):m*n,4) = T(1:m,1); %don't move if up is selected

%left column
T(1:m:m*(n-1)+1,5) = T(1:m:m*(n-1)+1,1); %don't move if left is selected

%right column
T(m:m:n*m,3) = T(m:m:m*n,1); %don't move if right is selected

%rewards information (column 2 flags exit states)
reward = zeros(N,2);
reward(46,:) = [100,1];
reward([9 25 27 43 44 45 47 48 49],:) = repmat([-100,1],9,1);
reward(7,:) = [10,1];
reward([11 13 22 36 42],:) = repmat([10,1],5,1);
reward([15 18 28 34],:) = repmat([-10,1],4,1);


for loopvar = 1:5000
% Grid world exploration
Loc = 1; 
L = Loc;
Score = 0;
done = 0;

%Number of exploration trials
if loopvar == 500
    epsilon = 0;
end;


while(done == 0)
    
    % Learning Algorithm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    [maxq,maxind] = sort(QValues(L,2:5),'descend'); 
    R = rand()
    % epsilon * 0.33 percent chance to take 3rd best action
    if R<(epsilon/3)
        a = maxind(3) + 1;
    % 1 - epsilon percent chance to take best action
    elseif R>epsilon
        a = maxind(1) + 1;
    % epsilon * 0.67 percent chance to take 2nd best action
    else
        a = maxind(2) + 1;
    end;
    % If action is known to be bad, take best action
    if(QValues(L,a) < -5)
        a = maxind(1) + 1;
    end;
    A = a;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %resolve action
    R = rand()
    if R<noise/2
        A = a+1
    elseif (R>noise/2)&&(R<noise)
        A = a-1;
    end;
        
    if A==6
        A=2;
    elseif A==1
        A=5;
    end
    
    %evaluate new location
    Loc = T(L,A);
    
    
    %determine rewards
    Update = -p + reward(Loc,1);
    Score = Score + Update;
    Q(L,A) = -p + gamma*reward(Loc,1);
    
    % Record information from experience
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TD = Update + gamma * max(QValues(Loc,2:5)) - QValues(L,A);
    QValues(L,A) = QValues(L,A) + rate * TD;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    L=Loc;
    done = reward(Loc,2);
    
end
totalscore = totalscore + Score;
if(Score > 10)
    post = post+1;
end;
end

Score = Score