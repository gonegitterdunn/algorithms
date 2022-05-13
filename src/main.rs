use core::num;
use std::collections::{BinaryHeap, HashMap};
use std::env;
use std::hash::Hash;
use std::iter::TakeWhile;
use std::str;
use std::str::Chars;

fn main() {
    // let target = env::args().skip(1).next().unwrap();
    // let string_to_check = env::args().skip(2).next().unwrap();

    // let target = target.chars();
    // let string_to_check_as_chars = string_to_check.chars();

    // // hashmap of given string to check pemutations of
    // let target_map = get_map_for_chars(target);

    // let initial_working_map = get_initial_working_map(string_to_check_as_chars, target_map.len());

    // println!("{:?}", target_map);
    // println!("{:?}", initial_working_map);

    // println!("{}", roman_to_int("XIV".into()));
    // println!("{:?}", fizz_buzz(26));
    // println!("{:?}", number_of_steps(123));
    // println!(
    //     "{}",
    //     maximum_wealth_2(vec![[1, 2, 3].to_vec(), [3, 2, 3].to_vec()])
    // )
    // println!("{}", can_construct("a;kfakljhawe".into(), "sdfga".into()));
    // println!(
    //     "{:?}",
    //     k_weakest_rows(
    //         [
    //             [1, 0, 0, 0].to_vec(),
    //             [1, 1, 1, 1].to_vec(),
    //             [1, 0, 0, 0].to_vec(),
    //             [1, 0, 0, 0].to_vec(),
    //         ]
    //         .to_vec(),
    //         2
    //     )
    // );

    // let a = ListNode::new(5);

    // let mut b = ListNode::new(4);
    // b.next = Some(Box::new(a));
    // let mut c = ListNode::new(3);
    // c.next = Some(Box::new(b));
    // let mut d = ListNode::new(2);
    // d.next = Some(Box::new(c));
    // let mut e = ListNode::new(1);
    // e.next = Some(Box::new(d));

    // println!("{}", is_palindrome(Some(Box::new(d))));

    // println!("{:?}", middle_node(Some(Box::new(d))));

    // println!("{}", search(vec![-1, 0, 3, 5, 9, 12], 9));

    // println!("{}", search_insert(vec![1, 3, 5, 6], 2));
    // println!("{:?}", sorted_squares_easy(vec![-7, -3, 2, 3, 11]));
    // println!("{:?}", sorted_squares_two_pointers(vec![-7, -3, 2, 3, 11]));
    // rotate_with_loop(&mut vec![1, 2, 3, 4, 5, 6, 7], 3);
    // println!("{:?}", rotate_with_enumerate(&mut vec![1, 2, 3, 4, 5, 6, 7], 3));
    // println!(
    //     "{:?}",
    //     rotate_with_split_at(&mut vec![1, 2, 3, 4, 5, 6, 7], 3)
    // );
    // println!("{:?}", rotate_with_split_at(&mut vec![-1], 2));
    // println!("{:?}", move_zeroes(&mut vec![0]));
    // println!("{:?}", move_zeroes_smarter(&mut vec![0, 1, 0, 3, 12]));
    // println!("{:?}", move_zeroes_smartest(&mut vec![0, 1, 0, 3, 12]));
    // println!(
    //     "{:?}",
    //     two_sum(
    //         vec![
    //             12, 13, 23, 28, 43, 44, 59, 60, 61, 68, 70, 86, 88, 92, 124, 125, 136, 168, 173,
    //             173, 180, 199, 212, 221, 227, 230, 277, 282, 306, 314, 316, 321, 325, 328, 336,
    //             337, 363, 365, 368, 370, 370, 371, 375, 384, 387, 394, 400, 404, 414, 422, 422,
    //             427, 430, 435, 457, 493, 506, 527, 531, 538, 541, 546, 568, 583, 585, 587, 650,
    //             652, 677, 691, 730, 737, 740, 751, 755, 764, 778, 783, 785, 789, 794, 803, 809,
    //             815, 847, 858, 863, 863, 874, 887, 896, 916, 920, 926, 927, 930, 933, 957, 981,
    //             997, 1000
    //         ],
    //         542
    //     )
    // );
    // println!("{:?}", two_sum(vec![2, 7, 11, 15], 9));
    // println!("{}", count_odds(8, 10));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_odds_two_odd_numbers() {
        assert_eq!(count_odds(3, 7), 3);
    }

    #[test]
    fn test_count_odds_two_even_numbers() {
        assert_eq!(count_odds(2, 10), 4);
    }

    #[test]
    fn test_count_odds_one_odd_one_even_number() {
        assert_eq!(count_odds(2, 11), 5);
    }
}

pub fn count_odds(low: i32, high: i32) -> i32 {
    if low % 2 == 0 && high % 2 == 0 {
        return (high - low) / 2;
    } else {
        return ((high - low) / 2) + 1;
    }
}

pub fn two_sum_different(numbers: Vec<i32>, target: i32) -> Vec<i32> {
    let mut first: usize = 0;
    let mut second: usize = numbers.len() - 1;
    let mut count = 0;

    while first < second {
        count += 1;
        if numbers[second] > target {
            second -= 1;
            continue;
        }
        if target - numbers[second] < numbers[first] {
            second -= 1;
        }

        if numbers[first] + numbers[second] == target {
            break;
        } else if numbers[first] + numbers[second] < target {
            first += 1;
        } else {
            second -= 1;
        }
    }
    println!("{}", count);
    return Vec::from([first as i32 + 1, second as i32 + 1]);
}

pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
    let mut left: usize = 0;
    let mut right: usize = numbers.len() - 1;
    while numbers[left] + numbers[right] != target {
        // eliminate all numbers greater than the target
        if numbers[right] > target {
            right -= 1;
        }
        // eliminate all right numbers that are too big for current left numbers
        else if target - numbers[right] < numbers[left] {
            right -= 1;
            continue;
        }

        if numbers[right] + numbers[left] > target {
            right -= 1;
        } else {
            left += 1;
        }
    }

    vec![(left + 1) as i32, (right + 1) as i32]
}

pub fn move_zeroes_smartest(nums: &mut Vec<i32>) {
    let mut first_zero: usize = 0;

    for index in 0..nums.len() {
        if nums[index] != 0 {
            nums.swap(index, first_zero);
            first_zero += 1;
        }
    }
    println!("{:?}", nums);
}

pub fn move_zeroes_smarter(nums: &mut Vec<i32>) {
    let mut first_zero: usize = 0;
    let mut index: usize = 0;
    while index < nums.len() {
        if nums.get(index).unwrap().ne(&0) {
            nums.swap(index, first_zero);
            first_zero += 1;
        }
        index += 1;
    }
    println!("{:?}", nums);
}

pub fn move_zeroes(nums: &mut Vec<i32>) {
    let mut left: usize = 0;
    let mut right = nums.len() - 1;
    while left < right as usize {
        if nums.get(left as usize).unwrap().eq(&0) {
            nums.remove(left as usize);
            nums.push(0);
            right -= 1;
        } else {
            left += 1;
        }
    }
    println!("{:?}", nums);
}

pub fn rotate_with_split_at(nums: &mut Vec<i32>, k: i32) {
    let split = nums.split_at(nums.len() % k as usize);
    println!("{:?}", split);
    *nums = [split.1, split.0].concat();

    println!("{:?}", nums);
}

pub fn rotate_with_enumerate(nums: &mut Vec<i32>, k: i32) {
    let len = nums.len();

    for (index, value) in nums.clone().iter().enumerate() {
        nums[(index + k as usize) % len] = *value;
    }
    println!("{:?}", nums);
}

pub fn rotate_with_loop(nums: &mut Vec<i32>, k: i32) {
    for _ in 0..k {
        nums.insert(0, *nums.last().unwrap());
        nums.pop();
    }
    println!("{:?}", nums);
}

pub fn sorted_squares_easy(nums: Vec<i32>) -> Vec<i32> {
    let mut nums_sq = nums.clone();

    nums_sq = nums_sq.iter().map(|x| x * x).collect();
    nums_sq.sort();
    nums_sq
}

pub fn sorted_squares_two_pointers(nums: Vec<i32>) -> Vec<i32> {
    let mut left = 0;
    let mut right = nums.len() - 1;

    let mut answer = Vec::<i32>::new();
    while left < right {
        if nums[left].abs() > nums[right].abs() {
            answer.push(nums[left] * nums[left]);
            left += 1
        } else {
            answer.push(nums[right] * nums[right]);
            right -= 1;
        }
    }
    answer.push(nums[left] * nums[right]);
    answer.into_iter().rev().collect()
}

pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    // if target is less than 1st element
    if target < nums[0] {
        return 0;
    }
    // if target is greater than last element
    if &target > nums.last().unwrap() {
        return nums.len() as i32;
    }
    // if nums's length is 1
    if nums.len() == 1 {
        if nums[0] <= target {
            return 0;
        } else {
            return 1;
        }
    }

    let mut low = 0;
    let mut high = nums.len() - 1;

    while low < high {
        let mid = (low + high) / 2;

        if target == nums[mid] {
            return mid as i32;
        } else if target < nums[mid] {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low as i32;
}

pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if head.as_ref().unwrap().next.eq(&None) {
        return head;
    }
    let mut head_mut = head;

    let mut head_mut_to_count = head_mut.as_ref();
    let mut length: i32 = 0;
    while let Some(node) = head_mut_to_count {
        head_mut_to_count = node.next.as_ref();
        length += 1;
    }

    for _ in 0..length / 2 {
        head_mut = head_mut.unwrap().next;
    }

    head_mut
}

pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
    if head.as_ref().unwrap().next == None {
        return true;
    }

    let mut mut_head = head;
    let mut values = Vec::<i32>::new();

    while let Some(node) = mut_head {
        values.push(node.val);
        mut_head = node.next;
    }

    let mut reversed_values = values.clone();
    reversed_values.reverse();

    values.eq(&reversed_values)
}

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub fn k_weakest_rows(mat: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
    let mut soldier_count: Vec<i32> = Vec::new();
    let matrix_iter = mat.into_iter();
    let mut count = 0;
    // sum up the soldiers in the matrixes
    for matrix in matrix_iter {
        soldier_count.insert(count, matrix.into_iter().sum());
        count += 1;
    }

    let mut soldier_count_dedup = soldier_count.clone();
    soldier_count_dedup.sort();
    soldier_count_dedup.dedup();

    // create a hashmap of (index of vec in vec, # soldiers)
    let mut index: i32 = 0;
    // let mut soldiers_per_index: HashMap<i32, i32> = HashMap::new();
    let mut index_per_soldiers: Vec<(i32, i32)> = Vec::new();
    for soldier in &soldier_count {
        // soldiers_per_index.insert(index, *soldier);
        index_per_soldiers.push((*soldier, index));
        index += 1;
    }
    index = 0;

    index_per_soldiers.sort_by(|a, b| a.0.cmp(&b.0));

    // sort soldier_count vec on soldier values
    soldier_count.sort();

    let mut answer = Vec::new();
    for item in &index_per_soldiers {
        answer.insert(index as usize, item.1);
        index += 1;
    }

    answer[0 as usize..k as usize].to_vec()
}

fn roman_to_int(s: String) -> i32 {
    let mut value: i32 = 0;
    let mut roman_numeral_stream = s.chars().peekable();

    while let Some(current_char) = roman_numeral_stream.next() {
        let next_char = roman_numeral_stream.peek();
        value += match (current_char, next_char) {
            ('I', Some('V')) => -1,
            ('I', Some('X')) => -1,
            ('X', Some('L')) => -10,
            ('X', Some('C')) => -10,
            ('C', Some('D')) => -100,
            ('C', Some('M')) => -100,
            ('I', _) => 1,
            ('V', _) => 5,
            ('X', _) => 10,
            ('L', _) => 50,
            ('C', _) => 100,
            ('D', _) => 500,
            ('M', _) => 1000,
            _ => panic!("Invalid character"),
        }
    }
    value
}

fn fizz_buzz(n: i32) -> Vec<String> {
    let mut answer = Vec::new();
    let fizzbuzz: String = "FizzBuzz".into();
    let buzz: String = "Buzz".into();
    let fizz: String = "Fizz".into();

    let mut count: i32 = 1;
    while count <= n {
        let element: String;

        if count % 15 == 0 {
            element = fizzbuzz.clone();
        } else if count % 5 == 0 {
            element = buzz.clone();
        } else if count % 3 == 0 {
            element = fizz.clone();
        } else {
            element = count.to_string();
        };

        answer.push(element);
        count = count + 1;
    }
    answer
}

pub fn number_of_steps(num: i32) -> i32 {
    match num == 0 {
        true => 0,
        false => match num % 2 == 0 {
            true => return number_of_steps(num / 2) + 1,
            false => return number_of_steps(num - 1) + 1,
        },
    }
}

pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
    let mut account_totals: Vec<i32> = Vec::new();

    let mut index: usize = 0;
    for val in accounts.iter() {
        let a: i32 = val.into_iter().sum();

        account_totals.insert(index, a);
        index += 1;
    }

    account_totals.sort();
    *account_totals.last().unwrap()
}

pub fn maximum_wealth_2(accounts: Vec<Vec<i32>>) -> i32 {
    let mut biggest: i32 = 0;
    for x in accounts.into_iter() {
        let sum: i32 = x.iter().sum();
        if sum > biggest {
            biggest = sum;
        }
    }
    biggest
}

pub fn maximum_wealth_3(accounts: Vec<Vec<i32>>) -> i32 {
    accounts.iter().map(|c| c.iter().sum()).max().unwrap_or(0)
}

pub fn can_construct(ransom_note: String, magazine: String) -> bool {
    // let a: &str = &ransom_note[..]
    let ransom_vec = ransom_note.as_str().chars().collect::<Vec<char>>();
    let magazine_vec = magazine.as_str().chars().collect::<Vec<char>>();

    let random_hashmap = ransom_vec
        .iter()
        .fold(HashMap::new(), |mut acc, character| {
            *acc.entry(character).or_insert(0) += 1;
            acc
        });

    let magazine_hashmap = magazine_vec
        .iter()
        .fold(HashMap::new(), |mut acc, character| {
            *acc.entry(character).or_insert(0) += 1;
            acc
        });

    for ch in ransom_vec.iter() {
        if random_hashmap.get(&ch) > magazine_hashmap.get(&ch) {
            return false;
        }
    }
    true
}

fn get_initial_working_map(inital: Chars, len: usize) -> HashMap<char, u16> {
    let string_to_check_inital: String = inital.take(len).collect();

    get_map_for_chars(string_to_check_inital.chars())
}

fn get_map_for_chars(target: Chars) -> HashMap<char, u16> {
    let mut target_map: HashMap<char, u16> = HashMap::new();

    for character in target.into_iter() {
        let counter = target_map.entry(character).or_insert(0);
        *counter += 1;
    }

    target_map
}

pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    if nums[0] == target {
        return 0;
    } else if nums[nums.len() - 1] == target {
        return (nums.len() - 1) as i32;
    } else {
        let target = target;
        let mut lower = 0;
        let mut upper = nums.len();

        while lower < upper {
            let middle = (lower + upper) / 2;
            if target == nums[middle] {
                return middle as i32;
            } else if nums[middle] > target {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }
        -1
    }
}
