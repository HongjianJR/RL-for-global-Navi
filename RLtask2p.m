% task 2
clc
clear all
load ('D:/qeval.mat'); % reward 100*4

     gamma =  0.9; % discount rate
        t_trials = 3000; % total number of trials
        s = ones(t_trials,1); % state
        a = ones(t_trials,1); % action
        time = zeros(1,10); % execution time
        optimal_policy = zeros(100,10); % which action to choose at every state, for 10 runs
        total_reward = zeros(1,10);
        max_reward = 0;
        o_p = 0; % state-changing policy
        for run = 1:10 % 10 runs
            tic
            Q = zeros(100,4);
            Q2 = ones(100,4); % stores the Q table for the previous trial
            N = zeros(100,4);
            for trial = 1:t_trials % each run has max 3000 trials
                k=1;
                alpha = 1; % initial meaningless number
                while k < 300 && alpha >= 0.005 % each trial max 300 moves 
                 
                            alpha = 500/(500+k);
                  
                    epsilon = 100/(100+k);
                    %%  epsilon greedy
                    temp = find(qevalreward(s(k),:)~=-1); % idx for valid actions
                    [re,t]=max(Q(s(k),temp));% optimal action
                    % choose exploit or explore
                    r=rand;
                    x=(r<epsilon); % 1 for exploit, 0 for explore
                    if x == 1   % exploit
                        a(k) = temp(t);
                    else        % explore
                        temp2 = find(qevalreward(s(k),:)~=qevalreward(s(k),temp(t)) & qevalreward(s(k),:)~=-1); % idx of other options
                        if(length(temp2))>0 % check if there are other options
                            a(k) = randi(length(temp2));
                            a(k) = temp2(a(k));
                        else % if not, select the only option
                            a(k) = temp(t);
                        end
                    end
                    %% move to the next step, update Q table
                    s(k+1) = move(s(k), a(k));
                    Q(s(k),a(k)) = Q(s(k),a(k))+alpha*(qevalreward(s(k),a(k))+gamma*max(Q(s(k+1),:))-Q(s(k),a(k))); % update Q function
                    N(s(k),a(k)) = N(s(k),a(k)) + 1;
                    if s(k+1) == 100
                        break
                    end
                   k = k+1;
                end 
              if sum(sum(abs(Q2-Q)))<1e-20 % if the update is too small, break
                  break
              end
                   Q2 = Q;  
            end % end one trial
            [~,optimal_policy(:,run)] = max(Q,[],2); % calculate policy
            [total_reward(run),p] = getreward(optimal_policy(:,run), qevalreward); % calculate reward
            if ~isnan(total_reward(run)) % if optimal policy found
                time(run) = toc;
            end
            if total_reward(run)>max_reward % compare reward with previous run
                o_p = p;
                max_reward = total_reward(run);
            end
        end
        %% show results
        if sum(~isnan(total_reward))>0
            fprintf('gamma = %.2f, %d runs reach the goal, average time = %.2fs, reward = %d\n', gamma, sum(~isnan(total_reward)), mean(time(time~=0)),max(total_reward));
            fprintf('The optimal policy is: \n');
             for i = 1:length(o_p)-1 
               fprintf('%d ->',o_p(i));
           end
            fprintf('%d\n',o_p(i+1));
              for i = 1:length(o_p)-1 
               if(o_p(i+1)==o_p(i)+1)
                   plot(mod(o_p(i),10)+0.5,-((o_p(i)-1)/10+1)+0.5,'>')
               end
               if(o_p(i+1)==o_p(i)+10)
                   plot(mod(o_p(i)+0.5,10),-((o_p(i)-1)/10+1)+0.5,'v')
               end
               if(o_p(i+1)==o_p(i)-1)
                   plot(mod(o_p(i),10)+0.5,-((o_p(i)-1)/10+1)+0.5,'<')
               end
               if(o_p(i+1)==o_p(i)-10)
                   plot(mod(o_p(i),10)+0.5,-((o_p(i)-1)/10+1)+0.5,'^')
               end
          hold on    
              end
        grid on
        set(gca,'XTick',[0:1:11])
        set(gca,'YTick',[-11:1:0])
    
            %plot_path(total_reward,optimal_policy);
        else
            fprintf('gamma = %.2f, optimal policy not found\n', gamma);
     
               end
            

function new_s = move(s,a)
action = [-1,10,1,-10];
new_s = s+action(a);
end

function [total_reward,move_policy] = getreward(optimal_policy, qevalreward)
n = 0;
state = 1;
move_policy = state;
total_reward = 0;
while( state ~= 100 && n < 100)
    action = optimal_policy(state);
    total_reward = total_reward + qevalreward(state, action);
    state = move(state, action);
    n = n+1;
    move_policy = [move_policy;state];
end
if state ~= 100 % never reached the goal
    total_reward = NaN;
end
end
