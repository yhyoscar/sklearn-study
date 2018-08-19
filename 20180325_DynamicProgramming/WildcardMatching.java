/**
	'?' Matches any single character.
	'*' Matches any sequence of characters (including the empty sequence).
	
	The matching should cover the entire input string (not partial).
	
	The function prototype should be:
	bool isMatch(const char *s, const char *p)
	
	Some examples:
	isMatch("aa","a") → false
	isMatch("aa","aa") → true
	isMatch("aaa","aa") → false
	isMatch("aa", "*") → true
	isMatch("aa", "a*") → true
	isMatch("ab", "?*") → true
	isMatch("aab", "c*a*b") → false
 */

public class WildcardMatching {
    public boolean isMatch ( String str, String pattern ) {
        int rows = str.length() + 1;
        int cols = pattern.length() + 1;
        
        boolean[][] grid = new boolean[rows][cols];
        grid[0][0] = true;
        
        // str with empty pattern, always no match
        for ( int i=1; i<rows; i++ ) {
            grid[i][0] = false;
        }
        
        // As long as the prefix are always *, then always match
        for ( int i=1; i<cols; i++ ) {
            if ( pattern.charAt(i-1) != '*' )
                break;
            grid[0][i] = true;
        }
        
        for ( int i=1; i<rows; i++ ) {
            for ( int j=1; j<cols; j++ ) {
                char a = str.charAt ( i-1 );
                char b = pattern.charAt ( j-1 );
                if ( a == b || b == '?' ) {
                    grid[i][j] = grid[i-1][j-1]; // current characters match, look back
                } else if ( b == '*' ) {
                    grid[i][j] = grid[i-1][j] || grid[i][j-1]; // remove star OR remove the character keep the star
                } else {
                    grid[i][j] = false;
                }
            }
        }
        
        return grid[rows-1][cols-1];
    }
}