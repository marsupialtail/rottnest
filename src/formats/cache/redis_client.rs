use crate::lava::error::LavaError;
use redis::aio::MultiplexedConnection;
use std::{
    env,
    ops::{Deref, DerefMut},
};

#[derive(Debug, Clone)]
pub struct RedisConnection {
    conn: MultiplexedConnection,
}

impl Deref for RedisConnection {
    type Target = MultiplexedConnection;

    fn deref(&self) -> &Self::Target {
        &self.conn
    }
}

impl DerefMut for RedisConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.conn
    }
}

impl RedisConnection {

    pub async fn get_ranges(&mut self, filename: &str) -> Result<Vec<(usize, usize)>, LavaError> {
        let key = format!("{}:range", filename);
        let res: Vec<u8> = redis::cmd("GET")
            .arg(key)
            .query_async(self.deref_mut())
            .await?;
        let ranges = bincode::deserialize(&res)?;
        Ok(ranges)
    }

    pub async fn set_ranges(
        &mut self,
        filename: &str,
        ranges: &Vec<(usize, usize)>,
    ) -> Result<(), LavaError> {
        let key = format!("{}:range", filename);
        let bytes = bincode::serialize(ranges)?;
        redis::cmd("SET")
            .arg(key)
            .arg(bytes)
            .query_async(self.deref_mut())
            .await?;
        Ok(())
    }

    pub async fn get_data(
        &mut self,
        filename: &str,
        from: u64,
        to: u64,
    ) -> Result<Vec<u8>, LavaError> {
        let key = format!("{}:{}:{}", filename, from, to);
        let res: Vec<u8> = redis::cmd("GET")
            .arg(key)
            .query_async(self.deref_mut())
            .await?;
        Ok(res)
    }

    pub async fn set_data(
        &mut self,
        filename: &str,
        from: u64,
        to: u64,
        data: Vec<u8>,
    ) -> Result<(), LavaError> {
        let key = format!("{}:{}:{}", filename, from, to);
        redis::cmd("SET")
            .arg(key)
            .arg(data)
            .query_async(self.deref_mut())
            .await?;
        Ok(())
    }
}

pub async fn get_redis_connection() -> Result<RedisConnection, LavaError> {
    let host = env::var("REDIS_HOST").unwrap_or("127.0.0.1".to_string());
    let port = env::var("REDIS_PORT")
        .unwrap_or("6379".to_string())
        .parse::<u16>()
        .map_err(|e| LavaError::Parse(format!("{:?}", e)))?;
    let client = redis::Client::open(format!("redis://{}:{}", host, port))?;
    let conn = client.get_multiplexed_tokio_connection().await?;
    Ok(RedisConnection { conn })
}

mod tests {
    use super::*;

    #[tokio::test]
    async fn test_redis_connection() {
        let mut conn = get_redis_connection().await.unwrap();
        let ranges = vec![(0, 10), (10, 20), (20, 30)];
        conn.set_ranges("test", &ranges).await.unwrap();
        let res = conn.get_ranges("test").await.unwrap();
        assert_eq!(res, ranges);
    }

    #[tokio::test]
    async fn test_redis_data() {
        let mut conn = get_redis_connection().await.unwrap();
        let data = vec![1, 2, 3, 4, 5];
        conn.set_data("test", 0, 5, data.clone()).await.unwrap();
        let res = conn.get_data("test", 0, 5).await.unwrap();
        assert_eq!(res, data);
    }

    #[tokio::test]
    async fn test_redis_key_non_exist() {
        let mut conn = get_redis_connection().await.unwrap();
        let res = conn.get_data("test_non_exists_key", 0, 5).await.unwrap();
        assert_eq!(0, res.len());
    }
}
