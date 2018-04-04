/**
	Given two words word1 and word2, find the minimum number of steps required to make word1 and word2 the same, where in each step you can delete one character in either string.
	
	Example 1:
	Input: "sea", "eat"
	Output: 2
	Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
 */

public class DeleteOperationForTwoStrings {
	public int minDistance(String word1, String word2) {
        if ( word1.isEmpty() )
            return word2.length();
        if ( word2.isEmpty() )
            return word1.length();    
            
        int rows = word1.length() + 1;
        int cols = word2.length() + 1;
        
        int[][] grid = new int[rows][cols];
        for ( int i=1; i<rows; i++ ) // no word2 present, need to delete every character form word1
            grid[i][0] = i;
        for ( int i=1; i<cols; i++ ) // no word1 present, need to delete every character from word2
            grid[0][i] = i;
        
        for ( int i=1; i<rows; i++ ) {
            for ( int j=1; j<cols; j++ ) {
                if ( word1.charAt(i-1) == word2.charAt(j-1) ) { // This two character match
                    grid[i][j] = grid[i-1][j-1];
                } else {
                    grid[i][j] = Math.min ( grid[i-1][j] + 1, grid[i][j-1] + 1 ); // Delete either character, and add one cost
                }
            }
        }
        
        return grid[rows-1][cols-1];
    }
}
