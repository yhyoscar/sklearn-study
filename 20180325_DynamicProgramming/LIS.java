import java.util.ArrayList;
import java.util.List;

/**
 	Given an unsorted array of integers
 		1. Find the length of longest increasing subsequence.
 		2. Find such a sequence
 */
public class LIS {
	
	public static int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = 1; 												// itself
        int ans = 1;
        for ( int i=1; i<nums.length; i++ ) {
            int maxSoFar = 1;
            for ( int j=0; j<i; j++ ) {
                if ( nums[i] > nums[j] ) {
                    maxSoFar = Math.max ( maxSoFar, dp[j] + 1 );
                }
            }
            dp[i] = maxSoFar;
            ans = Math.max ( ans, maxSoFar );
        }
        return ans;
    }
	
	public static List<Integer> lis ( int[] nums ) {
        int[] dp = new int[nums.length];
        int[] pre = new int[nums.length]; // Previous Element's Index in nums array
        dp[0] = 1;
        pre[0] = -1;
        int ans = 1;
        int longestEndIndex = 0;
        for ( int i=1; i<nums.length; i++ ) {
            int longest = 1;
            pre[i] = -1;
            for ( int j=0; j<i; j++ ) {
                if ( nums[i] > nums[j] && dp[j] + 1 > longest) {
                	longest = dp[j] + 1;
                	pre[i] = j;
                }
            }
            dp[i] = longest;
            if ( dp[i] > ans ) {
            	ans = dp[i];
            	longestEndIndex = i;
            }
        }
        
        // Trace Back based on previous indexes
        List<Integer> lis = new ArrayList<>();
        lis.add ( nums[longestEndIndex] );
        while ( pre[longestEndIndex] != -1 ) {
        	longestEndIndex = pre[longestEndIndex];
        	lis.add ( 0, nums[longestEndIndex] );
        }
        
        return lis;
    }
	
	public static void main ( String[] args ) {
		//int[] nums = new int[]{2, 4, 3, 5, 1, 7, 6, 9, 8};
		int[] nums = new int[]{10, 9, 2, 5, 3, 7, 101, 18};
		
		int length = lengthOfLIS ( nums );
		System.out.println ( "Length of LIS: " + length );
		
//		List<Integer> lis = lis ( nums );
//		for ( int num: lis ) {
//			System.out.print ( num + " " );
//		}
	}
}
