/**
 	Given two strings, find the longest common subsequence (LCS).
	
	For "ABCD" and "EDCA", the LCS is "A" (or "D", "C"), return 1.
	For "ABCD" and "EACB", the LCS is "AC", return 2.
 */
public class LCS {
	
	public static int longestCommonSubsequence(String A, String B) {
        int rows = A.length();
        int cols = B.length();
        
        int[][] matrix = new int[rows+1][cols+1];
        
        for ( int i=0; i<=rows; i++ ) // take 0 character from A
            matrix[i][0] = 0;
        for ( int i=0; i<=cols; i++ ) // take 0 character from B
            matrix[0][i] = 0;
            
        for ( int i=1; i<=rows; i++ ) {
            for ( int j=1; j<=cols; j++ ) {
                if ( A.charAt(i-1) == B.charAt(j-1) ) { // Match, plus 1
                    matrix[i][j] = matrix[i-1][j-1] + 1;
                } else {
                    matrix[i][j] = Math.max ( matrix[i-1][j], matrix[i][j-1] ); // No match, remove the last character from A or from B
                }
            }
        }
        
        return matrix[rows][cols];
    }
	
	public static void main ( String[] args ) {
		System.out.println ( longestCommonSubsequence("ABCD", "EACB") );
	}
}
