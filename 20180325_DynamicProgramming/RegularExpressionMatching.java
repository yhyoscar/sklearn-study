/**
	'.' Matches any single character.
	'*' Matches zero or more preceding element.
 	
	The matching should cover the entire input string (not partial).
	
	The function prototype should be:
	bool isMatch(const char *s, const char *p)
	
	Some examples:
	isMatch("aa","a") → false
	isMatch("aa","aa") → true
	isMatch("aaa","aa") → false
	isMatch("aaaa", "a*") → true
	isMatch("aa", ".*") → true
	isMatch("ab", ".*") → true
	isMatch("aab", "c*a*b") → true
 */

public class RegularExpressionMatching {
	public boolean isMatch ( String s, String p ) {
        int rows = s.length() + 1;
        int cols = p.length() + 1;
        
        boolean[][] valid = new boolean[rows][cols];
        valid[0][0] = true;
        
        for ( int i=1; i<cols; i++ ) {
            if ( p.charAt(i-1) == '*' )
                valid[0][i] = valid[0][i-2]; // A*B*C*... always match to empty string
        }
        
        for ( int i=1; i<rows; i++ ) {
            for ( int j=1; j<cols; j++ ) {
                char a = s.charAt ( i - 1 );
                char b = p.charAt ( j - 1 );
                if ( b == '.' || a == b ) {
                    valid[i][j] = valid[i-1][j-1];
                } else if ( b == '*' ) {
                    if ( p.charAt(j-2) == '.' || p.charAt(j-2) == a ) { // 1 or more occurrence of character before *
                        valid[i][j] = valid[i-1][j];
                    }
                    valid[i][j] = valid[i][j-2] || valid[i][j]; // 0 occurrence of character before *
                }
            }
        }
        
        return valid[rows-1][cols-1];
    }
}
