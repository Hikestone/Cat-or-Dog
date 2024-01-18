def score (player_name,call,actual_win) :
    print ("score of ",  player_name )
    
    if (call <= actual_win) :
        point = call*10
    else :
        point = call*10 + (actual_win - call)

    print("score of ",player_name," is ", point )

    return point

score(2,6,3)