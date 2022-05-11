use std::collections::HashMap;
use std::env;
use std::hash::Hash;
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

    let a = ListNode::new(1);

    let mut b = ListNode::new(1);
    b.next = Some(Box::new(a));
    let mut c = ListNode::new(2);
    c.next = Some(Box::new(b));
    let mut d = ListNode::new(1);
    d.next = Some(Box::new(c));

    println!("{}", is_palindrome(Some(Box::new(d))));
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
