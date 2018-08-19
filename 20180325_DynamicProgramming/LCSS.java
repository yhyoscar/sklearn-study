/**
	Given two strings, find the longest common substring.
	
	Given A = "ABCD", B = "CBCE", return 2.
 */
public class LCSS {
	public int longestCommonSubstring ( String A, String B ) {
        int rows = A.length();
        int cols = B.length();
        int ans = 0;
        
        int[][] matrix = new int[rows+1][cols+1];
        
        for ( int i=1; i<=rows; i++ ) {
        	for ( int j=1; j<=cols; j++ ) {
        		if ( A.charAt(i-1) == B.charAt(j-1) ) {
        			matrix[i][j] = matrix[i-1][j-1] + 1;  // find match, increase by 1
        			ans = Math.max ( ans, matrix[i][j] ); // Compare to find the so-far-best solution
        		} else {
        			matrix[i][j] = 0; // no match, meaning starting from index i & j, length of common substring will start from 0
        		}
        	}
        }
        
        return ans;
    }
}
