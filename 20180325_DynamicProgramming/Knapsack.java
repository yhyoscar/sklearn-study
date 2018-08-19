/**
 	If we have 4 items with size [2, 3, 5, 7], the backpack size is 11, we can select [2, 3, 5], so that the max size we can fill this backpack is 10. 
 	
 	If the backpack size is 12.  we can select [2, 3, 7] so that we can fulfill the backpack.
 */
public class Knapsack {
	
	public static int maxWeight ( int bagSize, int[] sizes ) {
		
		boolean[][] dp = new boolean[sizes.length + 1][bagSize + 1]; // f[i][j]: Can first i elements fit in size j

        dp[0][0] = true;
        
        for ( int i=1; i<sizes.length; i++ ) { // if bag size is 0, there are always a way to fill the bag with nothing
			dp[i][0] = true;
		}
		
		for ( int j=1; j<sizes.length; j++ ) { // if bag size is not 0 but no item, then you can never fill the bag
			dp[0][j] = false;
		}

        for ( int i = 1; i <= sizes.length; i++ ) {
            for ( int j = 1; j <= bagSize; j++ ) {
                dp[i][j] = dp[i - 1][j] || (j >= sizes[i - 1] && dp[i - 1][j - sizes[i - 1]]); // Pick iTH item or Not
            }
        }

        for ( int j = bagSize; j >= 0; j-- ) {
            if ( dp[sizes.length][j] ) {
                return j;
            }
        }

        return 0;
    }
	
	public static int maxValue ( int bagSize, int[] sizes, int[] prices ) {
		
		int[][] dp = new int[sizes.length + 1][bagSize + 1]; // f[i][j]: Max Value of first i elements filled in size j

		for ( int i=1; i<sizes.length; i++ ) { // if you bag is empty, your bag then always worth nothing
			dp[i][0] = 0;
		}
		
		for ( int j=1; j<sizes.length; j++ ) { // if you never put anything in the bag, your bag then always worth nothing
			dp[0][j] = 0;
		}
		
        for ( int i = 1; i <= sizes.length; i++ ) {
            for ( int j = 1; j <= bagSize; j++ ) {
            	if ( j - sizes[i-1] < 0 ) { // Item i is too heavy, cannot fit in the bag
            		dp[i][j] = dp[i-1][j];
            	} else {
            		dp[i][j] = Math.max ( dp[i-1][j], dp[i-1][j-sizes[i-1]] + prices[i-1] ); // Item i can fit, take the max value of either put it in the bag or not
            	}            
            }
        }

        return dp[sizes.length][bagSize];
    }
	
	public static void main ( String[] args ) {
		System.out.println ( maxWeight(100, new int[]{1,1,1,1,2,9}) );
	}
}
