public class EditDistance {
	
	/**
		Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)
		You have the following 3 operations permitted on a word:
	
		a) Insert a character (Insert a single character into pattern P to help it match text T, such as changing “ago” to “agog.”)
		b) Delete a character (Replace a single character from pattern P with a different character in text T, such as changing “shot” to “spot.”)
		c) Replace a character (Replace a single character from pattern P with a different character in text T, such as changing “shot” to “spot.”)
	**/
	
	public int minDistance(String word1, String word2, int costAdd, int costDelelte, int costReplace) {
        if ( word1.isEmpty() )
            return word2.length();
        if ( word2.isEmpty() )
            return word1.length();    
            
        int rows = word1.length() + 1;
        int cols = word2.length() + 1;
        
        int[][] grid = new int[rows][cols];
        
        grid[0][0] = 0;
        
        for ( int i=1; i<rows; i++ ) // Take 0 characters from B
            grid[i][0] = i;
        for ( int i=1; i<cols; i++ ) // Take 0 characters from A
            grid[0][i] = i;
        
        for ( int i=1; i<rows; i++ ) {
            for ( int j=1; j<cols; j++ ) {
                if ( word1.charAt(i-1) == word2.charAt(j-1) ) { // current characters match
                    grid[i][j] = grid[i-1][j-1];
                } else { // not match, then the best of insert/delete/replace + 1
                    grid[i][j] = Math.min ( Math.min(grid[i-1][j], grid[i][j-1]), grid[i-1][j-1] ) + 1;
                }
            }
        }
        return grid[rows-1][cols-1];
    }
}