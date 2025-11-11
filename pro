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





knows(alice,bob).
knows(bob,charlie).
knows(charlie,dana).

worksat(alice,techcorp).
worksat(bob,techcorp).
worksat(charlie,softsolutions).
worksat(dana,softsolutions).

programsin(alice,python).
programsin(dana,python).
programsin(bob,java).
programsin(bob,python).
programsin(charlie,cplusplus).

similarto(python,java).
similarto(python,cplusplus).
similarto(cplusplus,java).

connected(X,Y):-knows(X,Y).
connected(X,Y):-knows(Y,X).
connected(X,Y):-knows(X,Z),knows(Z,Y).

colleagues(X,Y):-worksat(X,C),worksat(Y,C),X\=Y.

skill(X,Y):-programsin(X,P),programsin(Y,P),X\=Y.

%query
colleagues(alice,Y).
skill(bob,Y).
skill(X,Y),worksat(X,C1),worksat(Y,C2),C1\=C2.
connected(charlie,Y).
knows(X,P),\+worksat(X,techcorp).




% Two people share the same hobby
same_hobby(X, Y) :-
    hobby(X, H),
    hobby(Y, H),
    X \= Y.

% Find employees who are parents
employee_parent(X, Company) :-
    works(X, Company),
    parent(X, _).

% People who play a sport and work at same company as someone
sport_and_same_company(X, Person) :-
    plays(X, _),
    works(X, Company),
    works(Person, Company),
    X \= Person.

% Connected through knows relationship
connected(X, Y) :- knows(X, Y).
connected(X, Y) :- knows(Y, X).

connected_to_both(X, P1, P2) :-
    connected(X, P1),
    connected(X, P2),
    P1 \= P2,
    X \= P1,
    X \= P2.

% Parent-child who share at least one hobby
parent_child_shared_hobby(P, C) :-
    parent(P, C),
    hobby(P, H),
    hobby(C, H).
