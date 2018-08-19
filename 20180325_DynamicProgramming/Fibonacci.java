public class Fibonacci {
	
	public static long fib_recursive ( int n ) {
		if ( n == 0 )
			return 0L;
		if ( n == 1 )
			return 1L;
		return fib_recursive(n-1) + fib_recursive(n-2);
	}
	
	public static long fib_dp ( int n ) {
		long[] results = new long[n+1];
		results[0] = 0;
		results[1] = 1;
		for ( int i=2; i<=n; i++ ) {
			results[i] = results[i-1] + results[i-2];
		}
		return results[n];
	}
	
	public static long fib_dp_v2 ( int n ) {
		long first = 0;
		long second = 1;
		if ( n == 0 )
			return first;
		if ( n == 1 )
			return second;
		long nth = -1;
		for ( int i=2; i<=n; i++ ) {
			nth = first + second;
			first = second;
			second = nth;
		}
		return nth;
	}
	
	public static void main ( String[] args ) {
		//System.out.println ( fib_recursive(50) );
		System.out.println ( fib_dp(10000000) );
	}
}


// 1000 - 15316 kb
// 10000000 - 93504 kb
// 78188 (kb) / 9999900 = 8.00653(kb) = 1 Long