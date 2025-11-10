% rules
% enrolledIn(User, Course)
enrolledIn(userA, deepLearningBasics).
enrolledIn(userB, pythonForDataScience).

% teaches(Instructor, Course)
teaches(instructor1, deepLearningBasics).
teaches(instructor1, pythonForDataScience).

% hasSkill(User, Skill)
hasSkill(userA, machineLearning).
hasSkill(userB, dataScience).
hasSkill(userC, machineLearning).

% friendOf(User1, User2)  (mutual)
friendOf(userA, userB).
friendOf(userB, userC).

% Because friendship is mutual
friendOf(X, Y) :- friendOf(Y, X).

% gaveFeedback(User, Course)
gaveFeedback(userA, deepLearningBasics).
gaveFeedback(userB, pythonForDataScience).

%rules
%two share a skill
shareSkill(U1, U2) :-
    hasSkill(U1, Skill),
    hasSkill(U2, Skill),
    U1 \= U2.

%Chain friendship (direct or indirect)
connected(U1, U2) :-
    friendOf(U1, U2).

connected(U1, U2) :-
    friendOf(U1, X),
    connected(X, U2),
    U1 \= U2.

%query
?- shareSkill(userC, X).
X = userA
?- teaches(instructor1, C), enrolledIn(userA, C).
C = deepLearningBasics
?- friendOf(userB, U), gaveFeedback(U, _).
U = userA
?- enrolledIn(U, deepLearningBasics), hasSkill(U, Skill).
U = userA,
Skill = machineLearning
?- connected(userA, X).
X = userB ;
X = userC


